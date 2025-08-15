import argparse
from airfoil_meshing import AirfoilMesher
from cfd_running import run_cfd_job
import os, shutil, pathlib
import click
import yaml
from box import Box

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', '-c', help="the path to the yaml config file", type=str, default='ros_config.yml' )
parser.add_argument('--run_folder', '-f', help="the directory where the results should be stored", type=str, default='runs/test_run' )
parser.add_argument('--plot', '-p', help="if the various plots should be shown, this will pause the process until plots are closed", action='store_true')
parser.add_argument('--overwrite', '-o', help="if the run folder should be overwritten", action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    config = Box.from_yaml(filename=os.path.abspath(args.config_file), Loader=yaml.FullLoader)
    run_folder = os.path.abspath(args.run_folder)
    plot = args.plot

    # overwrite protection
    if os.path.exists(run_folder):
        if args.overwrite or click.confirm('Do you want to overwrite {}?'.format(run_folder), abort=True):
            try:
                shutil.rmtree(run_folder)
            except Exception as e:
                print('Failed to overwrite %s. Reason: %s' % (run_folder, e))

    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=False)

    # create paths for meshes and cfd cases
    mesh_path = os.path.join(run_folder, 'mesh')
    run_path = os.path.join(run_folder, 'simulation')

    # mesh airfoil
    mesher = AirfoilMesher(foil_dat_file=config.airfoil_settings.airfoil_dat_file, mesh_name=mesh_path)
    ms = config.mesh_settings
    mesher.create_unstructured_mesh(use_base_coords = ms.use_base_coords, h_a = float(ms.h_a), h_0 = float(ms.h_0), R_b = float(ms.R_b), h_extrude = ms.h_extrude,
                                 refine_wake_len = float(ms.refine_wake_len), h_w = float(ms.h_w), view= args.plot)

    config['io_settings'] = {'mesh_path': mesh_path, 'init_from': 'uniform', 'run_path': run_path,'output_path': ''}

    # run the cfd
    run_cfd_job(config)
