from airfoils import Airfoil
from scipy import interpolate
import numpy as np
import gmsh
import ntpath
import matplotlib.pyplot as plt
from pebble import concurrent
import multiprocessing as mp
import tempfile
import os, shutil
import subprocess
try:
    from auxiliary_postpycess import read_grid, plot_grid
except:
    from .auxiliary_postpycess import read_grid, plot_grid
from distutils.dir_util import copy_tree
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner
from shapely.geometry import Polygon
import trimesh

class AirfoilMesher:
    def __init__(self, foil_dat_file=None, foil_coords = None, naca_foil = '4812', mesh_name = None):
        """
        A thin wrapper class that creates a 2D airfoil mesh and sets
        names for the surfaces of the boundaries.
        """
        self.mesh_name = mesh_name

        if foil_dat_file is not None:
            self.airfoil_coords = np.loadtxt(foil_dat_file, skiprows=1)
            if self.mesh_name is None:
                self.mesh_name = ntpath.basename(foil_dat_file)[:-4]

        elif foil_coords is not None:
            self.airfoil_coords = foil_coords
            if self.mesh_name is None:
                raise ValueError('Please enter a name for the mesh')

        else:
            foil = Airfoil.NACA4(naca_foil)
            x_points = np.concatenate([np.linspace(0.0, 0.01, 300),np.linspace(0.01, 1.0, 500)[1:]])
            x_points_reversed = list(reversed(x_points))[:-1]
            self.airfoil_coords = np.array([[*x_points_reversed, *x_points],
                                            [*foil.y_upper(x_points_reversed), *foil.y_lower(x_points)]]).T

            if self.mesh_name is None:
                self.mesh_name = 'naca_' + naca_foil

        
    def create_unstructured_mesh(self, use_base_coords = False, h_a = 5e-3, h_0 = 0.1, R_b = 100, h_extrude = 1.,
                             refine_wake_len = 0.0, h_w = 0.01, view= False):
        """
        Create the 2D unstructured mesh for simulating the airfoil.
        
        Arguments:
        use_base_coords   : (False) if False a spline fitting should be used to obtain coordinates
        h_a               : (5e-3) refinement size for airfoil surface
        h_0               : (0.1)  h-refinement size for overall sizing
        R_b               : the radius of the outer boundary.
        h_extrude         : (1.) openfoam needs an extrusion height. The geometry plane is extruded 
                            to a unit height in order to create cells.
        refine_wake_len   : (0.25) the length of the wake refinement line
        h_w               : (0.01) sizing of the refinement wake line
        """
    
        _gmsh = gmsh
        _gmsh.initialize()
        _gmsh.model.add(self.mesh_name)

        # points for the airfoil:
        if use_base_coords:
            coo = self.airfoil_coords
        else:
            t = np.linspace(0.0, 1.0, 1000)
            foil_spline = interpolate.splprep([self.airfoil_coords[:, 0], self.airfoil_coords[:, 1]], s=0.00002, k=3)
            coo = np.array(interpolate.splev(t, foil_spline[0], der=0)).transpose()

        # Create farfield circle first
        circle_center = _gmsh.model.occ.addPoint(0.5, 0.0, 0.0, h_0, 1)
        boundary = _gmsh.model.occ.addCircle(0.5, 0.0, 0.0, R_b)
        farfield_loop = _gmsh.model.occ.addCurveLoop([boundary])

        # Now create airfoil points with offset tag numbers
        tag_offset = 1000  # Start airfoil points at 1000 to avoid conflicts
        gmsh_airf_points = [_gmsh.model.occ.addPoint(coo[k,0], coo[k,1], 0, h_a, k + tag_offset) 
                        for k in range(len(coo))]
        n_airf_points = len(gmsh_airf_points)

        # create lines for airfoil:
        foil_curve = []
        for k in range(n_airf_points):
            k_start = gmsh_airf_points[k]
            k_end = gmsh_airf_points[(k + 1) % n_airf_points]
            line = _gmsh.model.occ.addLine(k_start, k_end)
            foil_curve.append(line)

        airfoil_loop = _gmsh.model.occ.addCurveLoop(foil_curve)

        # creating a circular surface with a foil-shaped hole
        base_surf = _gmsh.model.occ.addPlaneSurface([farfield_loop, airfoil_loop])
        _gmsh.model.occ.synchronize()

        # perform test to detect if mesh is broken
        @concurrent.process(context=mp.get_context('fork'), daemon=True, timeout=180)
        def test_mesh():
            _gmsh.model.mesh.generate(3)

        future = test_mesh()
        try:
            future.result()
        except:
            raise ValueError
        future.cancel()

        # add refinement line in the airfoil wake
        if refine_wake_len != 0.0:
            wake_offset = 2000  # Another offset for wake points
            p1 = _gmsh.model.occ.addPoint(1.05, 0, 0, h_w, wake_offset)
            p2 = _gmsh.model.occ.addPoint(1.05 + refine_wake_len, 0, 0, h_w, wake_offset + 1)
            refine_line = _gmsh.model.occ.addLine(p1, p2)
            _gmsh.model.occ.synchronize()
            _gmsh.model.mesh.embed(1, [refine_line], 2, base_surf)

        # extrude the surface and create farfield surface
        extruded = _gmsh.model.occ.extrude([(2,base_surf)], 0, 0, h_extrude, numElements=[1], recombine=True)
        _gmsh.model.occ.synchronize()

        # Get the volume and top surface from extrusion
        volume_tag = None
        top_surface = None
        lateral_surfaces = []
        
        # Process extruded entities in order
        for entity in extruded:
            dim, tag = entity
            if dim == 3:
                volume_tag = tag
            elif dim == 2:
                if not top_surface:
                    top_surface = tag
                else:
                    lateral_surfaces.append(tag)

        # First lateral surface is the farfield (from boundary circle)
        # Remaining surfaces are from the airfoil in order
        farfield_surface = lateral_surfaces[0]
        airfoil_surfaces = lateral_surfaces[1:]

        # definition of physical groups:
        front_surface = _gmsh.model.addPhysicalGroup(2, [top_surface])
        _gmsh.model.setPhysicalName(2, front_surface, 'FRONT')
        
        back_surface = _gmsh.model.addPhysicalGroup(2, [base_surf])
        _gmsh.model.setPhysicalName(2, back_surface, 'BACK')
        
        airfoil_group = _gmsh.model.addPhysicalGroup(2, airfoil_surfaces)
        _gmsh.model.setPhysicalName(2, airfoil_group, 'AIRFOIL')
        
        farfield_group = _gmsh.model.addPhysicalGroup(2, [farfield_surface])
        _gmsh.model.setPhysicalName(2, farfield_group, 'FARFIELD')
        
        volume_group = _gmsh.model.addPhysicalGroup(3, [volume_tag])
        _gmsh.model.setPhysicalName(3, volume_group, "internalField")

        # Mesh settings
        _gmsh.option.setNumber("Mesh.Algorithm", 6)
        _gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", h_0)
        _gmsh.option.setNumber('Mesh.MshFileVersion', 2.10)
        
        # Generate and optimize mesh
        _gmsh.model.mesh.generate(3)
        _gmsh.model.mesh.optimize("Netgen")
        _gmsh.write(f"{self.mesh_name}.msh")

        if view:
            _gmsh.fltk.run()
        
        _gmsh.finalize()

    def create_hyperbolic_mesh(self, use_base_coords,radius, j_max, y_plus, re_number, alpha,epsilon_implicit, epsilon_explicit,
                               farfield_uniformness, area_smoothing, te_spacing=1e-3,  view=False):
        # first parse the c2dext config dict from the inputs args
        options_dict = {
            "&SOPT": {
                "radi": radius,
            },
            "&VOPT": {
                "slvr": "'HYPR'",
                "topo": "'OGRD'",
                "jmax": j_max,
                "ypls": y_plus,
                "recd": re_number,
                "alfa": alpha,
                "epsi": epsilon_implicit,
                "epse": epsilon_explicit,
                "funi": farfield_uniformness,
                "asmt": area_smoothing,
            },
            "&OOPT": {
                "gdim": 2,
                "npln": 2,
                "dpln": 0.1,  # distance between planes
                "f3dm": "F"
            }
        }

        # points for the airfoil:
        if use_base_coords:
            coo = self.airfoil_coords
        else:
            t = np.linspace(0.0, 1.0, 400)
            foil_spline = interpolate.splprep([self.airfoil_coords[:, 0], self.airfoil_coords[:, 1]], s=0.00002, k=3)
            coo = np.array(interpolate.splev(t, foil_spline[0], der=0)).transpose()

        try:
            temp_dir = tempfile.mkdtemp()

            # write the temp config file
            with open(os.path.join(temp_dir, 'grid_options.in'), 'w') as options_file:
                for domain_key, options in options_dict.items():
                    options_file.write(domain_key + '\n')

                    for setting in options:
                        options_file.write(setting + " = " + str(options[setting]) + '\n')

                    options_file.write('/\n')

            # write the temp airfoil coords file
            with open(os.path.join(temp_dir, 'airfoil_coords.dat'), 'w') as coords_file:
                coords_file.writelines(os.path.basename(self.mesh_name).upper() + '\n')
                for i in range(len(coo)-1):
                    coords_file.writelines([str(coo[i, 0]) + '  ' + str(coo[i, 1]) + '\n'])
                # close the airfoil with the correct teg spacing
                teg_points = self.split_teg(coo[-1,:], coo[0,:], te_spacing)
                for i in range(len(teg_points)):
                    coords_file.writelines([str(teg_points[i, 0]) + '  ' + str(teg_points[i, 1]) + '\n'])

            additional_smoothing = 'n'  # This can be changed based on residual

            # run c2dext
            subprocess.run([os.path.join(os.path.dirname( __file__ ), 'bin', 'c2d-ext'), os.path.join(temp_dir, 'airfoil_coords.dat')],
                           input=''.join(['y\n GRID\n', "BUFF",
                                          '\n',
                                          additional_smoothing,
                                          '\n'
                                          'QUIT\n']),
                           encoding='ascii',
                           cwd=temp_dir)

            if view:
                imax, jmax, kmax, x, y, threed = read_grid(os.path.join(temp_dir, 'airfoil_coords.p3d'))
                plot_grid(x, y)

            # copy the blockmeshdict file to the correct location
            shutil.copy(os.path.join(temp_dir, 'airfoil_coords.blockMeshDict'), self.mesh_name + '.blockMeshDict')

        # clean up files
        finally:
            shutil.rmtree(temp_dir)

    def create_elliptical_mesh(self, use_base_coords,radius, j_max, y_plus, re_number, farfield_clustering, initial_smoothing,
                               final_smoothing, post_smoothing, normal_top, normal_bottom, te_spacing=1e-3,  view=False):
        # first parse the c2dext config dict from the inputs args
        options_dict = {
            "&SOPT": {
                "radi": radius,
                "fdst": farfield_clustering,
            },
            "&VOPT": {
                "slvr": "'ELLP'",
                "topo": "'OGRD'",
                "jmax": j_max,
                "ypls": y_plus,
                "recd": re_number,
                "stp1": initial_smoothing,
                "stp2": final_smoothing,
                "stp3": post_smoothing,
                "nrmt": normal_top,
                "nrmb": normal_bottom,
            },
            "&OOPT": {
                "gdim": 2,
                "npln": 2,
                "dpln": 0.1,  # distance between planes
                "f3dm": "F"
            }
        }

        # points for the airfoil:
        if use_base_coords:
            coo = self.airfoil_coords
        else:
            t = np.linspace(0.0, 1.0, 400)
            foil_spline = interpolate.splprep([self.airfoil_coords[:, 0], self.airfoil_coords[:, 1]], s=0.00002, k=3)
            coo = np.array(interpolate.splev(t, foil_spline[0], der=0)).transpose()

        try:
            temp_dir = tempfile.mkdtemp()

            # write the temp config file
            with open(os.path.join(temp_dir, 'grid_options.in'), 'w') as options_file:
                for domain_key, options in options_dict.items():
                    options_file.write(domain_key + '\n')

                    for setting in options:
                        options_file.write(setting + " = " + str(options[setting]) + '\n')

                    options_file.write('/\n')

            # write the temp airfoil coords file
            with open(os.path.join(temp_dir, 'airfoil_coords.dat'), 'w') as coords_file:
                coords_file.writelines(os.path.basename(self.mesh_name).upper() + '\n')
                for i in range(len(coo)-1):
                    coords_file.writelines([str(coo[i, 0]) + '  ' + str(coo[i, 1]) + '\n'])
                # close the airfoil with the correct teg spacing
                teg_points = self.split_teg(coo[-1,:], coo[0,:], te_spacing)
                for i in range(len(teg_points)):
                    coords_file.writelines([str(teg_points[i, 0]) + '  ' + str(teg_points[i, 1]) + '\n'])

            additional_smoothing = 'n'  # This can be changed based on residual

            # run c2dext
            subprocess.run([os.path.join(os.path.dirname( __file__ ), 'bin', 'c2d-ext'), os.path.join(temp_dir, 'airfoil_coords.dat')],
                           input=''.join(['y\n GRID\n', "BUFF",
                                          '\n',
                                          additional_smoothing,
                                          '\n'
                                          'QUIT\n']),
                           encoding='ascii',
                           cwd=temp_dir)

            if view:
                imax, jmax, kmax, x, y, threed = read_grid(os.path.join(temp_dir, 'airfoil_coords.p3d'))
                plot_grid(x, y)

            # copy the blockmeshdict file to the correct location
            shutil.copy(os.path.join(temp_dir, 'airfoil_coords.blockMeshDict'), self.mesh_name + '.blockMeshDict')

        # clean up files
        finally:
            shutil.rmtree(temp_dir)

    def create_snappy_hex_mesh(self, use_base_coords, radius=100, circular_layers=150, wi_cells=40,side_cells=20, view=False):

        # points for the airfoil:
        if use_base_coords:
            coo = self.airfoil_coords
        else:
            t = np.linspace(0.0, 1.0, 400)
            foil_spline = interpolate.splprep([self.airfoil_coords[:, 0], self.airfoil_coords[:, 1]], s=0.00002, k=3)
            coo = np.array(interpolate.splev(t, foil_spline[0], der=0)).transpose()

        try:
            temp_dir = tempfile.mkdtemp()

            # copy the template snappy case to the temp directory
            copy_tree(os.path.join(os.path.dirname(__file__), 'bin', 'snappy_hex_mesh_template'), temp_dir)

            # modify the blockMeshDict
            blockMeshDict = ParsedBlockMeshDict(os.path.join(temp_dir, 'system', 'blockMeshDict'))
            blockMeshDict["farfieldRad"] = radius
            blockMeshDict["negfarfieldRad"] = -radius
            blockMeshDict["farfieldCoord"] = radius/np.sqrt(2)
            blockMeshDict["negfarfieldCoord"] = -radius/np.sqrt(2)
            blockMeshDict["ycells"] = circular_layers
            blockMeshDict["wicells"] = wi_cells
            blockMeshDict["sidecells"] = side_cells
            blockMeshDict.writeFile()

            # modify the snappyHexMeshDict

            # run blockMesh
            blockMesh = BasicRunner(argv=["blockMesh", "-case", temp_dir], silent=True)
            blockMesh.start()

            # create airfoil STL file then run surfaceFeatures
            polygon = Polygon(coo)
            af_mesh = trimesh.creation.extrude_polygon(polygon, height=1)
            af_mesh.export(os.path.join(temp_dir, 'constant', 'triSurface', 'AIRFOIL.stl'))
            surfaceFeatures = BasicRunner(argv=["surfaceFeatures", "-case", temp_dir], silent=True)
            surfaceFeatures.start()

            # run snappyHexMeshDict
            snappyHexMesh = BasicRunner(argv=["snappyHexMesh", "-overwrite", "-case", temp_dir], silent=False)
            snappyHexMesh.start()

            # run extrudeMesh
            extrudeMeshDict = ParsedParameterFile(os.path.join(temp_dir, 'system', 'extrudeMeshDict'))
            extrudeMeshDict["sourceCase"] = '"{}"'.format(temp_dir)
            extrudeMeshDict.writeFile()
            extrudeMesh = BasicRunner(argv=["extrudeMesh", "-case", temp_dir], silent=True)
            extrudeMesh.start()

            # save the polyMesh in a zip file and copy it to the mesh folder
            shutil.make_archive(self.mesh_name, 'zip', os.path.join(temp_dir,'constant', 'polyMesh' ))

        # clean up files
        finally:
            pass
            # shutil.rmtree(temp_dir)

    def split_teg(self, start, end, spacing):
        num_segments = np.linalg.norm(start-end)//spacing
        x_delta = (end[0] - start[0]) / float(num_segments)
        y_delta = (end[1] - start[1]) / float(num_segments)
        points = []
        for i in range(1, int(num_segments)):
            points.append([start[0] + i * x_delta, start[1] + i * y_delta])
        return np.array([start] + points + [end])

if __name__ == '__main__':
    mesher = AirfoilMesher(foil_dat_file='aventa.dat', naca_foil = '0012', mesh_name = None)
    # mesher.create_unstructured_mesh(h_a =5e-3, h_0 = 0.07, R_b = 100, h_extrude = 0.1,
    #                                 refine_wake_len = 0.0, h_w = 0.01, use_base_coords=False, view=False)

    # mesher.create_hyperbolic_mesh(use_base_coords=False, radius= 100, j_max=200, y_plus=100, re_number=5e6, alpha=1,
    #                           epsilon_implicit=15.0, epsilon_explicit=0.0, farfield_uniformness=0.2, area_smoothing=40,
    #                           view=True)

    # mesher.create_elliptical_mesh(use_base_coords=False, radius= 100, j_max=200, y_plus= 30, re_number=5e6,
    #                               farfield_clustering=1,initial_smoothing=1000, final_smoothing=500, post_smoothing=0,
    #                               normal_top=10, normal_bottom=10, te_spacing=1e-3, view=True)

    mesher.create_snappy_hex_mesh(use_base_coords=False, circular_layers=100, wi_cells=20,side_cells=20)