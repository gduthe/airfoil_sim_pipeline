from .mesher import AirfoilMesher
import os, sys
from contextlib import contextmanager

def mesh_airfoil(shape, mesh_name, mesh_type='unstruct', Re=1e6, debug=False, plot=False):
    assert (mesh_type in ['unstruct','hyperbolic', 'elliptical', 'snappy'])
    if debug:  # debug mode
        mesher = AirfoilMesher(foil_coords=shape, mesh_name=mesh_name)
        if mesh_type == 'unstruct':
            mesher.create_unstructured_mesh(h_a=5e-3, h_0=0.13, R_b=100, h_extrude = 1.0,
                                            refine_wake_len=0.01, h_w=5e-2, use_base_coords=False, view=plot)
        elif mesh_type == 'hyperbolic':
            mesher.create_hyperbolic_mesh(use_base_coords=False, radius= 100, j_max=200, y_plus=30, re_number=Re, alpha=1,
                                          epsilon_implicit=15.0, epsilon_explicit=0.0, farfield_uniformness=0.2,
                                          area_smoothing=40, view=plot)
        elif mesh_type == 'elliptical':
            mesher.create_elliptical_mesh(use_base_coords=False, radius=100, j_max=200, y_plus=30, re_number=Re,
                                          farfield_clustering=1, initial_smoothing=1000, final_smoothing=500,
                                          post_smoothing=0, normal_top=10, normal_bottom=10, te_spacing=1e-3, view=plot)
        elif mesh_type == 'snappy':
            mesher.create_snappy_hex_mesh(use_base_coords=False, radius=100, circular_layers=100, wi_cells=20,
                                          side_cells=20, view=plot)

        return True

    else: # not debug mode, errors are excepted, stdout hidden
        try:
            print('Starting to mesh {}'.format(os.path.basename(mesh_name)))
            mesher = AirfoilMesher(foil_coords=shape, mesh_name=mesh_name)
            with stdout_redirected(to=mesh_name + '.logfile'):
                if mesh_type == 'unstruct':
                    mesher.create_unstructured_mesh(h_a=5e-3, h_0=0.13, R_b=100, h_extrude=1.0,
                                                    refine_wake_len=0.01, h_w=5e-2, use_base_coords=False, view=plot)
                elif mesh_type == 'hyperbolic':
                    mesher.create_hyperbolic_mesh(use_base_coords=False, radius=100, j_max=200, y_plus=30,
                                                  re_number=Re, alpha=1,  epsilon_implicit=15.0, epsilon_explicit=0.0,
                                                  farfield_uniformness=0.2, area_smoothing=40, view=plot)
                elif mesh_type == 'elliptical':
                    mesher.create_elliptical_mesh(use_base_coords=False, radius=100, j_max=200, y_plus=30,
                                                  re_number=Re, farfield_clustering=1, initial_smoothing=1000,
                                                  final_smoothing=500, post_smoothing=0, normal_top=10, normal_bottom=10,
                                                  te_spacing=1e-3, view=plot)
                elif mesh_type == 'snappy':
                    mesher.create_snappy_hex_mesh(use_base_coords=False, radius=100, circular_layers=100, wi_cells=20,
                                                  side_cells=20, view=plot)

            print('Successfully meshed {}'.format(os.path.basename(mesh_name)))
            return True
        except:
            print('Meshing of {} failed'.format(os.path.basename(mesh_name)))
            return False


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different