from .turbine_bc_generator import TurbineCfdBcGenerator
from .airfoil_bc_generator import AirfoilCfdBcGenerator
import numpy as np

def generate_bcs(bc_config:dict, num_samples:int, on_turbine=True, plot=False, sweep_mode=None, sweep_bounds=None):
    generator = TurbineCfdBcGenerator(bc_config) if on_turbine else AirfoilCfdBcGenerator(bc_config)
    output_dict = generator.generate_all_bcs(num_samples=num_samples, extra_uncertainty=False, plot=plot)
    
    # if sweep_mode is active (a variable name), all variables except the sweep variable are set to their mean value
    if sweep_mode is not None:
        for key in output_dict.keys():
            if key != sweep_mode:
                try:
                    output_dict[key] = output_dict[key]*0.0 + output_dict[key].mean()
                except:
                    output_dict[key] = [output_dict[key][0]] * num_samples
            else:
                if sweep_bounds is not None:
                    output_dict[key] = np.linspace(sweep_bounds[0], sweep_bounds[1], num_samples)


    bc_dict_list = []
    for i in range(num_samples):
        bc_dict = {}
        bc_dict['inflow_speed'] = output_dict['rel_u'][i].item() if on_turbine else output_dict['u'][i].item()
        bc_dict['reynolds_num'] = output_dict['Re'][i].item()
        bc_dict['angle_of_attack'] = output_dict['aoa'][i].item()
        bc_dict['kinematic_viscosity'] = output_dict['nu'][i].item()
        bc_dict['characteristic_length'] = 1
        bc_dict['turbulence_intensity'] = output_dict['ti'][i].item() / 100
        bc_dict['turbulence_length_scale'] = output_dict['tl'][i].item()

        suction_side_dict = {'roughness_extension': [-1, 0, -1, output_dict['rpl_s'][i].item()/100, 1, 2],
                              'roughness_height': output_dict['rh_s'][i].item(),
                             'roughness_density': 3, #hardcoded for now
                             'roughness_constant': 0.8}
        pressure_side_dict = {'roughness_extension': [-1, -1, -1, output_dict['rpl_p'][i].item() / 100, 0, 2],
                              'roughness_height': output_dict['rh_p'][i].item(),
                              'roughness_density': 3,  # hardcoded for now
                              'roughness_constant': 0.8}

        rh_cutoff = 100 #[um]

        if bc_config['airfoil_settings']['le_erosion'] is False:
             bc_dict['le_erosion'] = False
        else:
            if output_dict['rh_p'][i] < rh_cutoff and output_dict['rh_s'][i] < rh_cutoff:
                bc_dict['le_erosion'] = False
            else:
                bc_dict['le_erosion'] = True
                bc_dict['rough_patches'] = {}
                if output_dict['rh_p'][i] > rh_cutoff:
                    bc_dict['rough_patches'] ['AIRFOIL_Rough_Bot'] =  pressure_side_dict
                if output_dict['rh_s'][i] > rh_cutoff:
                    bc_dict['rough_patches'] ['AIRFOIL_Rough_Top'] = suction_side_dict
        bc_dict_list.append(bc_dict)

    return bc_dict_list