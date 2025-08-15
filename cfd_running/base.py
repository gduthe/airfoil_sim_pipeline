import shutil
import os
import numpy as np
import pandas as pd
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile, ParsedBoundaryDict
from PyFoam.RunDictionary.SolutionFile import SolutionFile
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.Applications.WriteDictionary import WriteDictionary
from PyFoam.Applications.Decomposer import Decomposer

def model_load(config):
    """
    Creates CurrentRun Case in tmp folder. Loads the template, mesh and solution/bc fields.
    """
    # Create CurrentRun Case in tmp folder.
    case = config.io_settings.run_path
    case_name = os.path.basename(case)
    if os.path.exists(case):
        shutil.rmtree(case)
    shutil.copytree(os.path.join(os.path.dirname( __file__ ), 'etc', '2D_foil_template'), case)

    #  Create mesh with blockMesh or gmsh
    if os.path.exists(config.io_settings.mesh_path + '.blockMeshDict'):
        shutil.copy(config.io_settings.mesh_path + '.blockMeshDict', os.path.join(case, 'system', 'blockMeshDict'))
        print(case_name + ": Making file from blockMesh")
        blockMesh = BasicRunner(argv=["blockMesh", "-case", case], silent=True)
        blockMesh.start()
    elif os.path.exists(config.io_settings.mesh_path + '.msh'):
        mesh_path = os.path.join(case, os.path.basename(os.path.normpath(case)) + '.msh')
        shutil.copy(config.io_settings.mesh_path+ '.msh', os.path.join(case, mesh_path))
        print(case_name + ": Converting gmsh mesh")
        gmshToFoam = BasicRunner(argv=["gmshToFoam", "-case", case, mesh_path], silent=True)
        gmshToFoam.start()

        boundaryDict = ParsedBoundaryDict(os.path.join(case, 'constant', 'polyMesh', 'boundary' ))
        boundaryDict["BACK"]['type'] = 'empty'
        boundaryDict["FRONT"]['type'] = 'empty'
        boundaryDict["AIRFOIL"]['type'] = 'wall'
        boundaryDict.writeFile()
    elif os.path.exists(config.io_settings.mesh_path + '.zip'):
        shutil.unpack_archive(config.io_settings.mesh_path + '.zip', os.path.join(case, 'constant', 'polyMesh'))
        print(case_name + ": Extracting snappyHexMesh")
    else:
        raise NotImplementedError('Other meshing capabilities not implemented yet')

    # check the mesh quality
    print(case_name + ": Checking mesh quality")
    checkMesh = BasicRunner(argv=["checkMesh", "-case", case], silent=True)
    checkMesh.start()
    # if mesh is not ok, throw an error
    with open(file=os.path.join(case, 'PyFoam.checkMesh.logfile')) as f:
        lines = f.readlines()
    if not any("Mesh OK." in s for s in lines):
        print(case_name + ": Mesh check failed")
        raise ValueError('Mesh check failed, simulation aborted')

    if config.bc_settings.le_erosion:
        print(case_name + ": Leading edge erosion activated. Creating rough patches.")
        split_patches(config, case)

    # Load solutions corresponding to mesh file:
    if config.io_settings.init_from == "uniform":
        print(case_name + ": Initializing with uniform field")
        foam_load_field(config)
        foam_set_fields(config)
    elif config.io_settings.init_from == "exact":
        print(case_name + ": Loading the solution")
        foam_load_field(config)
        # foam_set_bc(config)
    elif config.io_settings.init_from == "mappable":
        print(case_name + ": Similar solution exists. Mapping the solution")
        foam_load_field(config)
        foam_map_field(config)
        print(case_name + ": Similar solution exists. Mapping the solution")

    return


def model_configure(config):
    """
    Creates model configuration and sets it for a run
    turbulence_setup = case + "/constant/turbulenceProperties"
    methods_setup = case + "/system/fvSchemes"
    methods_control_setup = case + "/system/fvSolution"
    Note: key in fvSolution = {"simpleFoam":"SIMPLE","pisoFoam":"PISO","pimpleFoam":"PIMPLE"}
    """

    # RUN CONTROLS
    case = config.io_settings.run_path
    case_name = os.path.basename(case)
    run_control_setup = os.path.join(case, 'system', 'controlDict')
    WriteDictionary([run_control_setup, "application", config.cfd_settings['solution']['solver']])
    WriteDictionary([run_control_setup, "endTime", str(config.cfd_settings['max_iterations'])])
    WriteDictionary([run_control_setup, "deltaT", str(config.cfd_settings['time_step'])])
    WriteDictionary([run_control_setup, "writeInterval", str(config.cfd_settings['write_interval'])])

    turbulence_setup = os.path.join(case, 'constant', 'momentumTransport')
    WriteDictionary([turbulence_setup, "RAS['RASModel']", config.cfd_settings['turbulence_model']])

    # METHODS CONTROLS
    methods_dict = {"simpleFoam": "SIMPLE", "pimpleFoam": "PIMPLE", "pisoFoam": "PISO"}
    solver_method = methods_dict[config.cfd_settings['solution']['solver']]
    #  TODO use parsedfile option instead of writedictionary
    if config.cfd_settings['methods_control']:
        methods_control_setup = os.path.join(case, 'system', 'fvSolution')
        WriteDictionary([methods_control_setup,
                         solver_method + "['residualControl']['p']",
                         config.cfd_settings['convergence_criteria'][0]])
        WriteDictionary([methods_control_setup,
                         solver_method + "['residualControl']['U']",
                         config.cfd_settings['convergence_criteria'][1]])
        WriteDictionary([methods_control_setup,
                         solver_method + "['residualControl']['\"(k|epsilon|omega|ReThetat|gammaInt)\"']",
                         config.cfd_settings['convergence_criteria'][2]])


        WriteDictionary([methods_control_setup,
                         "relaxationFactors['fields']['p']",
                         config.cfd_settings['relaxation_factors'][0]])
        WriteDictionary([methods_control_setup,
                         "relaxationFactors['equations']['U']",
                         config.cfd_settings['relaxation_factors'][1]])
        WriteDictionary([methods_control_setup,
                         "relaxationFactors['equations']['\"nuTilda|nut|kt|kl|k|epsilon|omega|ReThetat|gammaInt\"']",
                         config.cfd_settings['relaxation_factors'][2]])

    return


def model_run(config):
    case = config.io_settings.run_path
    case_name = os.path.basename(case)
    # if config.cfd_settings['turbulence_model'] == "kOmegaSST":
    #     # print(case_name + ": BODGING kOmegaSST")
    #     # bodge_sst(case, config)
    print(case_name + ": Running potentialFOAM")
    potential_init(case)

    if config.cfd_settings['number_of_cores'] == 1:
        print(case_name + ": Running OpenFOAM")
        foam_run(config.cfd_settings['solution']['solver'], case)
    else:
        foam_run_parallel(config.cfd_settings['solution']['solver'], case,
                          config.cfd_settings['number_of_cores'])

    if config.cfd_settings['yPlus']:
        y_plus = BasicRunner(argv=[config.cfd_settings['solution']['solver'],
                                   "-case", case,
                                   "-postProcess -func yPlus"],
                             silent=True)
        y_plus.start()

    if config.bc_settings['le_erosion']:
        merge_patches(config, case)

    # calculateCp2D = BasicRunner(argv=["./cfd_running/etc/calculateCp2D",
    #                                   "-case", case,
    #                                   "-latestTime"],
    #                             silent=True)
    # calculateCp2D.start()

    return

def model_store(config):
    case = config.io_settings.run_path
    case_name = os.path.basename(case)
    post_process = SolutionDirectory(os.path.join(case, "postProcessing", "airfoil_coeffs"))
    post_process.latestDir()

    if config.cfd_settings['open_foam_fields']:
        # TODO Dump everything to output for now
        dire = SolutionDirectory(case, archive=config.io_settings.output_path)

        # Choose what logs are kept in the archive at the end of each run
        dire.addBackup("PyFoamSolve.logfile")
        dire.addBackup("PyFoamSolve.analyzed")
        dire.addBackup("postProcessing/airfoil_coeffs")
        dire.addBackup("postProcessing/yPlus")
        dire.addBackup("Cp.dat")
        dire.addBackup(dire.initialDir())


        # Archive name: uuid
        # TODO Note: it is possible to store dataset and push to parent the
        #  manifest with solved field files through config object...
        dire.lastToArchive(config.id)
        '''
        output_dataset = config.output_manifest.get_dataset("open_foam_fields")
        cp_data = Datafile(
            path=os.path.join(config.id, "U"),
            path_from=output_dataset,
            skip_checks=True,
            tags="useful:metadata",
            id=config.id
        )
        output_dataset.append(cp_data)
        '''

    # Store the coeffs
    # TODO check if the files exist!!!! (the force coef written at least once before convergence is achieved!)
    force_coeffs_data = pd.read_csv(os.path.join(case,"postProcessing","airfoil_coeffs",
                                                 post_process.latestDir(),"forceCoeffs.dat"),
                                        sep="\t", header=None, comment='#')
    cp_coeffs_data = pd.read_csv(os.path.join(case, "Cp.dat"), sep="\t", header=None, comment='#',
                                 names=["x-coord", "y-coord", "Cp"])

    # Return last time step coeff.
    # TODO Save last time-interval data in case of transient simulations

    config.output_values['aoa'] = config.input_values['angle_of_attack']

    config.output_values['cl'] = force_coeffs_data.iloc[-1, 3]
    config.output_values['cd'] = force_coeffs_data.iloc[-1, 2]
    config.output_values['cm'] = force_coeffs_data.iloc[-1, 2]
    config.output_values['cp'] = cp_coeffs_data.to_dict(orient="records")

    #  Delete CurrentRun
    print(case_name + ": Cleaning up temp")
    shutil.rmtree(os.path.join("data", "tmp", "CurrentRun"))


def foam_run(solver, case):
    # Run the solver
    run = ConvergenceRunner(BoundingLogAnalyzer(), argv=[solver, "-case", case], silent=True)
    run.start()
    return


def foam_run_parallel(solver, case, number_of_cores):
    case_name = os.path.basename(case)
    # Run the solver
    print(case_name + ": Decomposing Domain")
    Decomposer(args=[case, number_of_cores], silent=True)

    print(case_name + ": Running OpenFOAM")
    run = ConvergenceRunner(BoundingLogAnalyzer(),
                            argv=["mpirun -np", str(number_of_cores), solver, "-case", case, "-parallel"], silent=True)
    run.start()
    print(case_name + ": Reconstructing Domain")
    reconstruct = BasicRunner(argv=["reconstructPar", "-case", case], silent=True)
    reconstruct.start()

    return


def split_patches(config, case):
    # TODO unite split and merge patches into single method, with operation=split/merge argument
    topoSet_actions = []
    create_patches = []
    for patch_name in config.bc_settings['rough_patches'].keys():
        vertices = config.bc_settings['rough_patches'][patch_name]['roughness_extension']
        topoSet_actions.append({"name": patch_name,
                                "type": "faceSet",
                                "action": "new",
                                "source": "patchToFace",
                                "sourceInfo": {"name": "AIRFOIL"}})
        topoSet_actions.append({"name": patch_name,
                                "type": "faceSet",
                                "action": "subset",
                                "source": "boxToFace",
                                "sourceInfo": {"box": "(%f %f %f)(%f %f %f)"
                                                      % (vertices[0], vertices[1], vertices[2],
                                                         vertices[3], vertices[4], vertices[5])}})

        create_patches.append({"name": patch_name,
                               "patchInfo": {"type": "wall"},
                               "constructFrom": "set",
                               "set": patch_name})

    topoSetDict = ParsedParameterFile(os.path.join(case, 'system', 'topoSetDict'))
    topoSetDict["actions"] = topoSet_actions
    topoSetDict.writeFile()
    BasicRunner(argv=["topoSet", "-case", case], silent=True).start()

    createPatchDict = ParsedParameterFile(os.path.join(case, 'system', 'createPatchDict'))
    createPatchDict["patches"] = create_patches
    createPatchDict.writeFile()
    BasicRunner(argv=["createPatch", "-case", case, "-overwrite"], silent=True).start()
    #  Keeping the mesh for now.
    # shutil.copytree(os.path.join(case, "constant", "polyMesh"),
    #                 os.path.join("data","input","open_foam_mesh","2D", config.id))


def merge_patches(config, case):
    patches=["AIRFOIL"]
    patches.extend(config.bc_settings['rough_patches'].keys())
    create_patches = [{"name": "AIRFOIL",
                       "type": "faceSet",
                       "patchInfo": {"type": "wall"},
                       "constructFrom": "patches",
                       "patches": patches}]

    createPatchDict = ParsedParameterFile(os.path.join(case, 'system', 'createPatchDict'))
    createPatchDict["patches"] = create_patches
    createPatchDict.writeFile()
    BasicRunner(argv=["createPatch", "-case", case, "-overwrite"], silent=True).start()
    return


def foam_load_field(config):
    shutil.copytree(os.path.join(os.path.dirname( __file__ ),'etc','uniform_field_template'),
                    os.path.join(config.io_settings.run_path, '0'))
    return


def foam_set_fields(config):
    case = config.io_settings.run_path
    # Define solution director
    dire = SolutionDirectory(case)

    # Change initial BC to reflect input

    # U control
    u_initial = SolutionFile(dire.initialDir(), "U")

    # Setting nu
    transport_properties = ParsedParameterFile(os.path.join(dire.constantDir(), 'transportProperties'))
    transport_properties['nu'] = "nu [0 2 -1 0 0 0 0] " + str(config.bc_settings['kinematic_viscosity'])
    transport_properties.writeFile()

    # nuTilda control
    if config.cfd_settings['turbulence_model'] == 'SpalartAllmaras':
        chi = 4
        nutilda_initial = SolutionFile(dire.initialDir(), "nuTilda")
        nutilda_initial.replaceInternal(chi * config.bc_settings['kinematic_viscosity'])

    c_mu = 0.09

    epsilon_models = ['kEpsilon', 'realizableKE']
    omega_models = ['kOmega', 'kOmegaSST', 'kOmegaSSTLM', 'kkLOmega']

    # k control
    if config.cfd_settings['turbulence_model'] != 'SpalartAllmaras':
        turbulent_kinetic_energy = \
            3 / 2 * (config.bc_settings['turbulence_intensity'] * config.bc_settings['inflow_speed']) ** 2
        k_initial = SolutionFile(dire.initialDir(), "k")
        k_initial.replaceInternal(turbulent_kinetic_energy)

    # epsilon/omega control
    if config.cfd_settings['turbulence_model'] in epsilon_models:
        # In the OpenFOAM code l is defireturnreturnned as mixing length and therefore: (c_m**0.75), but here we use c_m
        # to initialise the inlet
        turbulent_kinetic_energy_dissipation = \
            (c_mu ** 0.75) * (turbulent_kinetic_energy ** 1.5) / config.bc_settings['turbulence_length_scale']
        epsilon_initial = SolutionFile(dire.initialDir(), "epsilon")
        epsilon_initial.replaceInternal(turbulent_kinetic_energy_dissipation)
    elif config.cfd_settings['turbulence_model'] in omega_models:
        # Same as with tke dissipation openfoam divides omega by (c_mu ** 0.25)
        # TODO This simplifies to 2.23*TI*inflow_speed/turbulence_length_scale, which in case of the farfield boundary
        #   should be in the range proposed by Menter:
        #   inflow_speed/Length_of_domain<omega_farfield<10*inflow_speed/Length_of_domain
        turbulence_specific_dissipation = \
            (turbulent_kinetic_energy ** 0.5) / ((c_mu ** 0.25) * config.bc_settings['turbulence_length_scale'])
        omega_initial = SolutionFile(dire.initialDir(), "omega")
        omega_initial.replaceInternal(turbulence_specific_dissipation)

        if config.cfd_settings['turbulence_model'] == 'kOmegaSSTLM':
            Tu = config.bc_settings['turbulence_intensity']*100
            if Tu <= 1.3:
                Re_thetat_i = 1173.51 - 589.428*Tu + 0.2196*Tu**(-2)
            else:
                Re_thetat_i = 331.50*(Tu - 0.5658)**(-0.671)
            ReThetat_initial = SolutionFile(dire.initialDir(), "ReThetat")
            ReThetat_initial.replaceInternal(Re_thetat_i)

    # nut control
    nut_initial = SolutionFile(dire.initialDir(), "nut")

    if config.cfd_settings['turbulence_model'] == 'SpalartAllmaras':
        nut = config.bc_settings['kinematic_viscosity'] * chi ** 4 / (chi ** 3 + 7.1 ** 3)
    elif config.cfd_settings['turbulence_model'] in epsilon_models:
        nut = c_mu * turbulent_kinetic_energy ** 2 / turbulent_kinetic_energy_dissipation
    elif config.cfd_settings['turbulence_model'] in omega_models:
        nut = turbulent_kinetic_energy / turbulence_specific_dissipation
    nut_initial.replaceInternal(nut)

    # Wall functions
    #-----------------------------------------------------------------------------------------
    #  TODO Introduce smooth wall functions controls

    # Roughness control

    if config.bc_settings['le_erosion']:

        kw_roughness_extension={nut_initial:"nutkRoughWallFunction",
                                k_initial:"kRoughWallFunction",
                                omega_initial:"omegaRoughWallFunction"}

        for field in kw_roughness_extension.keys():
            for patch_name in config.bc_settings['rough_patches'].keys():
                wall_bc = {"type": kw_roughness_extension[field],
                           "Ks": convert_roughness_to_ks(config.bc_settings['rough_patches'][patch_name]['roughness_height'],
                                                         config.bc_settings['rough_patches'][patch_name]['roughness_density']),
                           "Cs": config.bc_settings['rough_patches'][patch_name]['roughness_constant'],
                           "value": "$internalField"}

                bc_file = ParsedParameterFile(field.name)
                bc_file["boundaryField"][patch_name] = wall_bc
                bc_file.writeFile()


    Umag = config.bc_settings['inflow_speed']

    aoa = config.bc_settings['angle_of_attack']
    # Change farfield condition by modifying internal Field
    Ux = np.cos(np.radians(aoa)) * Umag
    Uy = np.sin(np.radians(aoa)) * Umag
    u_initial.replaceInternal("(%f %f 0)" % (Ux, Uy))

    # Super dirty way to manipulate dict, should use ParsedParameterFile
    # TODO clean up the mess
    #      this can be done only once for steady state using post-process

    option = "--strip-quotes-from-value"
    dict = case + "/system/forceCoeffsDict"
    aoa_rad = np.radians(aoa)
    liftDir = "(%f %f 0)" % (-np.sin(aoa_rad), np.cos(aoa_rad))
    dragDir = "(%f %f 0)" % (np.cos(aoa_rad), np.sin(aoa_rad))
    WriteDictionary([option, dict, "airfoil_coeffs['liftDir']", liftDir])
    WriteDictionary([option, dict, "airfoil_coeffs['dragDir']", dragDir])
    WriteDictionary([option, dict, "airfoil_coeffs['magUInf']", Umag])

    return config

def convert_roughness_to_ks(roughness_height, roughness_density):
    Rh = roughness_height
    Rd = roughness_density
    if Rh > 100:
        ks = (1.49783 * 10**-9*(9.29170684919976 *10**24 *Rh - 9.24939916631901 * 10**26)**(0.5) - 2950.72) * \
             (-0.768462 + 5.46075*10**-17 * (3.16961301221313 * 10**32 *Rd + 9.79025531790832 * 10**31)**(0.5))*1e-6
    else:
        ks = Rh*1e-6
    return ks

def foam_map_field(config):
    print('Mapping is not implemented yet')

def potential_init(case):
    potentialFoam = BasicRunner(argv=["potentialFoam", "-case", case], silent=True)
    potentialFoam.start()

def bodge_sst(case, config):
    run_control_setup = os.path.join(case, 'system', 'controlDict')
    turbulence_setup = os.path.join(case, 'constant', 'momentumTransport')
    WriteDictionary([run_control_setup, "endTime", "100"])
    WriteDictionary([run_control_setup, "writeInterval", "100"])
    WriteDictionary([turbulence_setup, "RAS['RASModel']", "kOmega"])
    foam_run(config.cfd_settings['solution']['solver'], case)
    WriteDictionary([run_control_setup, "writeInterval", str(config.cfd_settings['write_interval'])])
    WriteDictionary([turbulence_setup, "RAS['RASModel']", config.cfd_settings['turbulence_model']])
    WriteDictionary([run_control_setup, "endTime", str(config.cfd_settings['max_iterations'])])
