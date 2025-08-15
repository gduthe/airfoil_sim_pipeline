import argparse
from airfoil_generation import sample_catalog_airfoils, sample_specific_airfoils, generate_random_airfoil_shapes
from boundary_condition_generation import generate_bcs
from airfoil_meshing import mesh_airfoil
from cfd_running import run_cfd_job
from joblib import Parallel, delayed
import os, pathlib, shutil, sys
import click
import yaml
from box import Box
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--num_airfoils', '-na', help="the number of airfoils to simulate", type=int, default=10)
parser.add_argument('--num_sims', '-ns', help="the number of simulations to run per airfoil", type=int, default=1)
parser.add_argument('--same_bcs', '-sbc', help="if the boundary conditions should be the same for each airfoil shape", action='store_true')
parser.add_argument('--config_file', '-c', help="the path to the yaml config file", type=str, default='config.yml' )
parser.add_argument('--run_folder', '-f', help="the directory where the results should be stored", type=str)
parser.add_argument('--num_threads', '-nt', help="the number of threads to use in ", type=int, default=4)
parser.add_argument('--plot', '-p', help="if the various plots should be shown, this will pause the process until plots are closed", action='store_true')
parser.add_argument('--overwrite', '-o', help="if the run folder should be overwritten", action='store_true')
parser.add_argument('--on_turbine', '-ot', help="if turbine operating conditions should be mimicked", action='store_true')
parser.add_argument('--registry_file', '-rf', help="path to the registry file of previously simulated airfoils", type=str, default=None)
parser.add_argument('--sweep_mode', '-sm', help="Parameter to sweep (e.g., 'u', 'ti, 'aoa', 'Re', 'Ma', etc.)", type=str, default=None)
parser.add_argument('--sweep_bounds', '-sb', help="Tuple with the bounds for the parameter sweep (e.g., -25 25 for 'aoa')",nargs="+", type=float, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    num_airfoils = args.num_airfoils
    num_sims_per_airfoil = args.num_sims
    config_dict = Box.from_yaml(filename=os.path.abspath(args.config_file), Loader=yaml.FullLoader)
    run_folder = os.path.abspath(args.run_folder)
    if config_dict.run_settings.debug:
        num_threads = 1
    else:
        num_threads = args.num_threads
    plot = args.plot
    on_turbine = args.on_turbine
    registry_file = args.registry_file

    # overwrite protection
    if os.path.exists(run_folder):
        if args.overwrite or click.confirm('Do you want to overwrite {}?'.format(run_folder), abort=True):
            for filename in os.listdir(run_folder):
                file_path = os.path.join(run_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to overwrite %s. Reason: %s' % (file_path, e))

    # create meshes, simulations and etc folders
    meshes_folder = os.path.join(run_folder,'meshes')
    sims_folder = os.path.join(run_folder, 'simulations')
    etc_folder = os.path.join(run_folder, 'etc')
    pathlib.Path(meshes_folder).mkdir(parents=True, exist_ok=False)
    pathlib.Path(sims_folder).mkdir(parents=True, exist_ok=False)
    pathlib.Path(etc_folder).mkdir(parents=True, exist_ok=False)

    # create paths for meshes and cfd cases
    mesh_paths = [os.path.join(meshes_folder, 'mesh_' + str(i).zfill(len(str(num_airfoils)))) for i in range(num_airfoils)]
    run_paths = [os.path.join(sims_folder, 'simulation_'+str(i).zfill(len(str(num_sims_per_airfoil*num_airfoils)))) for i in range(num_sims_per_airfoil*num_airfoils)]

    # load registry if provided
    if registry_file and os.path.exists(registry_file):
        registry_df = pd.read_csv(registry_file)
        previously_simulated_airfoils = set(registry_df['airfoil_names'])
    else:
        previously_simulated_airfoils = set()

    # modify airfoil generation to consider the registry
    if config_dict.airfoil_settings.generation_type == 'catalog':
        shapes_list, airfoil_names = sample_catalog_airfoils(
            catalog_path=config_dict.airfoil_settings.gen_path,
            num_airfoils=num_airfoils,
            plot_shapes=plot,
            previously_simulated_airfoils=previously_simulated_airfoils
        )
    elif config_dict.airfoil_settings.generation_type == 'random':
        shapes_list = generate_random_airfoil_shapes(
            run_dir=config_dict.airfoil_settings.gen_path,
            num_airfoils=num_airfoils,
            plot_shapes=plot
        )
        airfoil_names = [f"random_airfoil_{i}" for i in range(len(shapes_list))]
    else:
        raise NotImplementedError()

    # handle shorter than expected shape lists
    if len(shapes_list) == 0:
        print("No new airfoils to simulate. Exiting.")
        sys.exit(0)
    elif len(shapes_list) < num_airfoils:
        num_airfoils = len(shapes_list)
        print(f"Only {num_airfoils} new airfoils left to simulate. Adjusting number of airfoils.")
    
    # update registry with newly simulated airfoils
    if registry_file:
        new_registry_data = pd.DataFrame({'airfoil_names': airfoil_names})
        if os.path.exists(registry_file):
            registry_df = pd.concat([registry_df, new_registry_data]).drop_duplicates()
        else:
            registry_df = new_registry_data
        registry_df.to_csv(registry_file, index=False)

    # create a dataframe with the airfoil names
    names_df = pd.DataFrame({'airfoil_names': airfoil_names})
    names_df.to_csv(os.path.join(etc_folder, 'airfoil_names.csv'))

    # generate a bunch of input conditions
    num_samples = num_sims_per_airfoil if args.same_bcs else num_sims_per_airfoil*num_airfoils
    bc_dict_list = generate_bcs(config_dict, num_samples=num_samples, plot=plot, on_turbine=on_turbine, 
                                sweep_mode=args.sweep_mode, sweep_bounds=args.sweep_bounds)
    bc_dict_list = [bc_dict for _ in range(num_airfoils) for bc_dict in bc_dict_list ] if args.same_bcs else bc_dict_list

    # create the list of cfd config dicts, includes mesh locations
    cfd_configs = []
    for i, bc_dict in enumerate(bc_dict_list):
        cfd_config = config_dict.copy()
        cfd_config['bc_settings'] = bc_dict
        cfd_config['run_settings'] = {'run_id': True}
        cfd_config['io_settings'] = {'mesh_path': mesh_paths[i // num_sims_per_airfoil], 'init_from': 'uniform','run_path': run_paths[i], 'output_path': ''}

        cfd_configs.append(cfd_config)

    # mesh shapes in parallel and save the meshes
    mesh_success_list = Parallel(n_jobs=num_threads)(delayed(mesh_airfoil)(shape, mesh_paths[i], config_dict.mesh_settings.mesh_type,
                                                                   bc_dict_list[i]['reynolds_num'], config_dict.run_settings.debug) for i, shape in enumerate(shapes_list))
    mesh_success_list = [success for success in mesh_success_list for _ in range(num_sims_per_airfoil)]
    shape_list = [shape for shape in shapes_list for _ in range(num_sims_per_airfoil)]
    airfoil_names = [name for name in airfoil_names for _ in range(num_sims_per_airfoil)]

    # run the cfd in parallel over multiple threads, each handling distribution of cores
    cfd_success_list = Parallel(n_jobs=num_threads)(delayed(run_cfd_job)(config, mesh_success_list[i], config_dict.run_settings.debug) for i, config in enumerate(cfd_configs))

    # write all etc information
    etc_df = pd.DataFrame({'shapes': shape_list, 'airfoil_names': airfoil_names, 'bcs':bc_dict_list, 'mesh_success':mesh_success_list, 'cfd_success':cfd_success_list})
    etc_df.to_csv(os.path.join(etc_folder, 'run_info.csv'))
    config_dict.to_yaml(os.path.join(etc_folder, 'run_config.yml'))