from .base import *
from box import Box

class ConvergenceError(Exception):
    """Raised when solution diverges"""
    pass


def run_cfd(config):
    """
    Runs the application
    """
    config = Box(config)
    if config.bc_settings['inflow_speed'] is None:
        config.bc_settings['inflow_speed'] = config.bc_settings['reynolds_num']*config.bc_settings['kinematic_viscosity']

    print(os.path.basename(config.io_settings.run_path) + ': Starting CFD run')
    model_load(config)
    model_configure(config)

    try:
        model_run(config)
    except:
        print("The solution failed for case {}".format(config))
        return
    print(os.path.basename(config.io_settings.run_path) + ': Run complete')
    config.to_yaml(os.path.join(config.io_settings.run_path, 'config.yml'))
