import numpy as np
from scipy.stats import qmc, weibull_min, norm, cauchy, halfnorm
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import seaborn as sns

class TurbineCfdBcGenerator:
    def __init__(self, config_dict:dict):
        """
        Boundary condition generator for turbine operational conditions.
        """

        self.inflow_settings = config_dict['inflow_settings']
        self.airfoil_settings = config_dict['airfoil_settings']
        self.turbine_settings = config_dict['turbine_settings']

        self.sampler = qmc.Sobol(d=4, scramble=True)

        # plotting settings
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2


    def __gen_wind_velocities(self, sobol_samples:np.array, plot=False):
        # compute scale param and get shape param
        a = 2 * self.inflow_settings['mean_V'] / np.sqrt(np.pi)
        k_V = self.inflow_settings['k_V']

        # calculate the probabilities corresponding to the bounds
        bounds = np.array([self.turbine_settings['cutin_u'], self.turbine_settings['cutout_u']])
        P_bounds = weibull_min.cdf(bounds, k_V, loc=0, scale=a)

        # scale and shift samples to cover P_bounds
        x = P_bounds[0] + sobol_samples*(P_bounds[1]-P_bounds[0])

        # compute inflow velocity
        u = weibull_min.ppf(x, k_V, scale=a)

        # get turbine parameters
        max_rot_rpm = self.turbine_settings['max_rot_rpm']
        min_rot_rpm = self.turbine_settings['min_rot_rpm']
        rated_u = self.turbine_settings['rated_u']
        cutin_u = self.turbine_settings['cutin_u']

        rot_rpm_slope = (max_rot_rpm - min_rot_rpm) / (rated_u - cutin_u)
        rot_rpm_intercept = max_rot_rpm - (max_rot_rpm-min_rot_rpm)/(rated_u-cutin_u)*rated_u
        rot_rpm = rot_rpm_slope*u+rot_rpm_intercept
        rot_rpm[u>=rated_u] = max_rot_rpm
        rot_rpm[u<cutin_u] = 0

        # relative wind speed, allow some variability due to rotor azimuth
        v_rot = (self.airfoil_settings['blade_span_location']*self.turbine_settings['rotor_diameter']/2)*rot_rpm*2*np.pi/60
        mean_rel_u = np.sqrt(v_rot**2+u**2) #ignoring induction and blade flapping, tower motion, etc.
        rel_u = np.random.normal(loc=mean_rel_u, scale=0.03*mean_rel_u)

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            sns.histplot(u, ax=axs[0], stat='density')
            sns.kdeplot(u, ax=axs[0], color='red', clip=bounds)
            axs[0].set_xlabel('Wind Speed [m/s]')
            axs[0].set_title('Free Wind Speed Distribution')
            sns.scatterplot(x=u, y=rel_u, ax=axs[1])
            axs[1].set_xlabel('Free Wind Speed [m/s]')
            axs[1].set_ylabel('Relative Wind Speed [m/s]')
            axs[1].set_title('Relative Wind Speed vs. Free Wind Speed')
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return u, rel_u

    def __gen_turbulence(self, u:np.array, plot=False):
        # generate free stream turbulence and turbulence intensity
        mu_T = self.inflow_settings['Iref'] * (0.75* u + 3.8)
        sig_T = 1.4 * self.inflow_settings['Iref']
        m = np.log((mu_T**2) / np.sqrt(sig_T**2 + mu_T**2))
        v = np.sqrt(np.log((sig_T/ mu_T)** 2 + 1))
        turb = np.random.lognormal(mean=m, sigma=v)
        ti = 100 * turb/ u

        # generate turbulence length scale
        mu_TL = self.inflow_settings['mean_turb_Lscale']*(0.75*u + 3.6)
        sig_TL = 1.4 * self.inflow_settings['Iref']
        m = np.log((mu_TL**2)/ np.sqrt(sig_TL**2 + mu_TL**2))
        v = np.sqrt(np.log((sig_TL / mu_TL)**2 + 1))

        # Turbulence length similar to equation 15 in https://wes.copernicus.org/articles/3/533/2018/
        du_dz = 0.2 #assume constant positive exponent
        tl = self.turbine_settings['height_above_ground']* np.random.lognormal(mean=m, sigma=v)/ 100.00 / du_dz

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
            sns.scatterplot(x=u, y=turb, ax=axs[0])
            axs[0].set_xlabel('Wind Speed [m/s]')
            axs[0].set_ylabel('Turbulence [m/s]')
            axs[0].set_title('Turbulence vs. Wind Speed')
            sns.scatterplot(x=u, y=ti, ax=axs[1])
            axs[1].set_xlabel('Wind Speed [m/s]')
            axs[1].set_ylabel('Turbulence Intensity [%]')
            axs[1].set_title('Turbulence Intensity vs. Wind Speed')
            sns.scatterplot(x=u, y=tl, ax=axs[2])
            axs[2].set_xlabel('Wind Speed [m/s]')
            axs[2].set_ylabel('Turbulence Length [% Chord L]')
            axs[2].set_title('Turbulence Length vs. Wind Speed')
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return turb, ti, tl

    def __gen_freestream_shear(self, u:np.array, plot=False):
        mu_alpha = 0.088 * (np.log(u) - 1)
        sig_alpha = 1.0 / u
        shearExp = np.random.normal(loc=mu_alpha, scale=sig_alpha)

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            sns.scatterplot(x=u, y=shearExp, ax=ax)
            ax.set_xlabel('Wind Speed [m/s]')
            ax.set_ylabel('Shear Exponent [m/s]')
            ax.set_title('Shear Exponent vs. Wind Speed')
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')

        return shearExp

    def __gen_air_properties(self,sobol_samples:np.array, plot=False):
        qL = -20 # lower bound
        qH = 50 # Upper bound
        bounds = np.array([qL, qH])

        # calculate the probabilities corresponding to the bounds
        mean_temp = self.inflow_settings['mean_air_temp']
        COV_temp = self.inflow_settings['COV_air_temp']
        P_bounds = norm.cdf(bounds, loc=mean_temp, scale=COV_temp * mean_temp)

        # scale and shift samples to cover P_bounds
        x = P_bounds[0] + sobol_samples * (P_bounds[1] - P_bounds[0])

        # variability in air temperature in Kelvin
        airtemp = 273 + norm.ppf(x, mean_temp, COV_temp * mean_temp)

        # mean air density as a function of temperature
        frho = -0.0043 * (airtemp) + 2.465

        # variability in air density
        rho = np.random.normal(loc=frho, scale=frho * 0.01)

        # air viscosity(dynamic) as a function of temperature
        fmu = 5E-08*airtemp + 4E-06

        # mu
        mu_mu = fmu
        sig_mu = fmu * self.inflow_settings['COV_Dyn_Viscosity']
        m = np.log((mu_mu**2)/np.sqrt(sig_mu**2 + mu_mu**2))
        v = np.sqrt(np.log((sig_mu/mu_mu)**2 + 1))
        mu = np.random.lognormal(mean=m, sigma=v)

        if plot:
            # first figure - air density plots
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
            sns.histplot(airtemp, ax=axs[0,0], stat='density')
            sns.kdeplot(airtemp, ax=axs[0,0], color='red')
            axs[0,0].set_xlabel('Temperature [deg. K]')
            axs[0,0].set_title('Temperature Distribution')
            sns.scatterplot(x=airtemp, y=frho, ax=axs[0,1])
            axs[0,1].set_xlabel('Temperature [deg. K]')
            axs[0,1].set_ylabel('Air Density [kg/m3]')
            axs[0,1].set_title('Mean Air Density vs. Temperature')
            sns.histplot(rho, ax=axs[1, 0], stat='density')
            sns.kdeplot(rho, ax=axs[1, 0], color='red')
            axs[1,0].set_xlabel('Air Density [kg/m3]')
            axs[1,0].set_title('Air Density Distribution')
            sns.scatterplot(x=airtemp, y=rho, ax=axs[1,1])
            axs[1,1].set_xlabel('Temperature [deg. K]')
            axs[1,1].set_ylabel('Air Density [kg/m3]')
            axs[1,1].set_title('Air Density vs. Temperature')
            for line in axs:
                for ax in line:
                    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

            # second figure - air viscosity plots
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
            sns.scatterplot(x=airtemp, y=fmu, ax=axs[0])
            axs[0].set_xlabel('Temperature [deg. K]')
            axs[0].set_ylabel('Air Viscosity (dynamic) [kg/m-s]')
            axs[0].set_title('Mean Air Viscosity vs. Temperature')
            sns.scatterplot(x=airtemp, y=mu, ax=axs[1])
            axs[1].set_xlabel('Temperature [deg. K]')
            axs[1].set_ylabel('Air Viscosity (dynamic) [kg/m-s]')
            axs[1].set_title('Air Viscosity vs. Temperature')
            sns.histplot(rho, ax=axs[2], stat='density')
            sns.kdeplot(rho, ax=axs[2], color='red')
            axs[2].set_xlabel('Air Viscosity (dynamic) [kg/m-s]')
            axs[2].set_title('Air Viscosity Distribution')
            fig.tight_layout(pad=3.0)
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')

        return airtemp, rho, mu

    def __gen_dimensionless_nums(self, rel_u:np.array, rho:np.array, mu:np.array, airtemp:np.array, u:np.array
                                 ,extra_uncertainty=False, plot=False):
        mean_Re = rho* rel_u* self.airfoil_settings['chord_length']/ mu

        if extra_uncertainty:
            sig_Re = mean_Re * self.inflow_settings['COV_Re']
            m = np.log((mean_Re ** 2) / np.sqrt(sig_Re ** 2 + mean_Re ** 2))
            v = np.sqrt(np.log((sig_Re / mean_Re) ** 2 + 1))
            Re = np.random.lognormal(mean=m, sigma=v)
        else:
            Re = mean_Re

        adiabatic_gas_const = 8.314 #J / mol.k
        specific_heat_ratio = 1.670 # specific heat capacity ratio at 20 deg.
        mol_mass_gas = 28.9e-3 # kg / mol
        mean_Ma = rel_u/np.sqrt(specific_heat_ratio*adiabatic_gas_const*airtemp / mol_mass_gas) # assuming ideal gas

        if extra_uncertainty:
            sig_Ma = mean_Ma * self.inflow_settings['COV_Ma']
            m = np.log((mean_Ma ** 2) / np.sqrt(sig_Ma ** 2 + mean_Ma ** 2))
            v = np.sqrt(np.log((sig_Ma / mean_Ma) ** 2 + 1))
            Ma = np.random.lognormal(mean=m, sigma=v)
        else:
            Ma = mean_Ma

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
            sns.scatterplot(x=u, y=Re, ax=axs[0])
            axs[0].set_xlabel('Wind Speed [m/s]')
            axs[0].set_ylabel('Reynolds Number [-]')
            axs[0].set_title('Reynolds Number vs. Wind Speed')
            sns.scatterplot(x=u, y=Ma, ax=axs[1])
            axs[1].set_xlabel('Wind Speed [m/s]')
            axs[1].set_ylabel('Mach Number [-]')
            axs[1].set_title('Mach Number vs. Wind Speed')
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return Re, Ma

    def __gen_angle_of_attack(self, u:np.array, ti:np.array, plot=False):
        # angle of attack not correlated with the spanwise location of the airfoil for now, to be done
        # ref: https://www.sciencedirect.com/science/article/pii/S0960148117307280

        # condition the angle on the wind speed using the following condition function
        ui = np.arange(3, 25,0.5)
        yri = 0.000010678892580* ui**6 - 0.000924777861724* ui**5 + 0.030660063307706* ui**4 - 0.478409474031769 * ui**3 + 3.403952380001092 * ui**2 - 8.573205022003162* ui + 5.734176802223377
        aoa_i = self.airfoil_settings['mean_ang_attack']* yri/ max(yri) # normalize
        mean_aoa = pchip_interpolate(ui, aoa_i, u)

        # create a Cauchy probability distribution object with degrees
        # location parameter equal to aoa, and sigma = 1 to set the scale parameter equal to 1.
        dampr = ti / max(ti)
        aoa = np.zeros_like(u)
        for i in range(len(u)):
            mu = mean_aoa[i]
            sigma = np.abs(mean_aoa[i] * self.airfoil_settings['COV_ang_attack'] * dampr[i])
            bounds = np.array([mean_aoa[i] - np.abs(mean_aoa[i]) * 0.75, mean_aoa[i] + np.abs(mean_aoa[i])  * 0.25])
            P_bounds = cauchy.cdf(bounds, loc=mu, scale=sigma)

            # scale and shift random sample to cover P_bounds
            x = P_bounds[0] + np.random.random() * (P_bounds[1] - P_bounds[0])

            # compute angle of attack
            aoa[i] = cauchy.ppf(x, loc=mu, scale=sigma)

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
            sns.scatterplot(x=u, y=aoa, ax=axs[0])
            axs[0].set_xlabel('Wind Speed [m/s]')
            axs[0].set_ylabel('Mean Angle of Attack [°]')
            axs[0].set_title('Mean Angle of Attack vs. Wind Speed')
            sns.scatterplot(x=u, y=aoa, ax=axs[1])
            axs[1].set_xlabel('Wind Speed [m/s]')
            axs[1].set_ylabel('Angle of Attack [°]')
            axs[1].set_title('Angle of Attack vs. Wind Speed')
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return aoa

    def __gen_roughness_properties(self, sobol_samples:np.array, plot=False):
        # grain roughness size is 1.3 times larger on pressure compared to suction side
        roughness_height_ratio = 1.3
        qL = 80 # lower bound of the grain roughness size
        qH = 350 # upper bound of the grain roughness size
        bounds = np.array([qL, qH])
        mu = self.airfoil_settings['mean_equ_sand_grain_rough_H']
        sigma = self.airfoil_settings['STD_sand_grain_rough_H']

        # calculate the probabilities corresponding to the bounds, then scale and shift random samples
        P_bounds = halfnorm.cdf(bounds, loc=mu, scale=sigma)
        x = P_bounds[0] + sobol_samples[:,0] * (P_bounds[1] - P_bounds[0])

        # equivalent sand grain roughness heights
        rh_p = halfnorm.ppf(x, loc=mu, scale=sigma)
        rh_s = halfnorm.ppf(x, loc=mu, scale=sigma) / roughness_height_ratio

        # patch length is 1.3 times larger on pressure compared to suction side
        patch_length_ratio = 1.3

        lL = 0 # lower bound of the Roughness patch length, percent of chord length
        lH = 13 # upper bound of the Roughness patch length on the pressure side, percent of chord length
        bounds = np.array([lL, lH])
        mu = self.airfoil_settings['mean_rough_patch_length']
        sigma = self.airfoil_settings['STD_rough_patch_length']

        # calculate the probabilities corresponding to the bounds, then scale and shift random samples
        P_bounds = halfnorm.cdf(bounds, loc=mu, scale=sigma)
        x = P_bounds[0] + sobol_samples[:,1]  * (P_bounds[1] - P_bounds[0])

        # rough patch lengths
        rpl_p = halfnorm.ppf(x, loc=mu, scale=sigma)
        rpl_s = halfnorm.ppf(x, loc=mu, scale=sigma) / patch_length_ratio

        # add labels for the erosion severity
        # based on https://m-selig.ae.illinois.edu/pubs/SareenSapreSelig-2014-WindEnergy-Erosion.pdf
        lee_labels = []
        for i in range(len(rh_p)):
            if rh_p[i]<0.5: # corresponds to pits
                lee_labels.append('Type_A')
            if rh_p[i]>=0.5 and rh_p[i]<2.5:  # corresponds to pits and gouges
                lee_labels.append('Type_B')
            if rh_p[i]>=2.5: # corresponds to delamination pits and gouges
                lee_labels.append('Type_C')

        if plot:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 12))
            sns.histplot(rh_p, ax=axs[0,0], stat='density')
            sns.kdeplot(rh_p, ax=axs[0,0], color='red', clip=bounds)
            axs[0,0].set_xlabel('Equivalent Sand Grain Roughness Height (pressure) [mm]')
            axs[0,0].set_title('Pressure Side ESGR Height Distribution')
            sns.histplot(rh_s, ax=axs[0,1], stat='density')
            sns.kdeplot(rh_s, ax=axs[0,1], color='red', clip=bounds)
            axs[0, 1].set_xlabel('Equivalent Sand Grain Roughness Height (suction) [mm]')
            axs[0, 1].set_title('Suction Side ESGR Height Distribution')
            sns.histplot(rpl_p, ax=axs[1,0], stat='density')
            sns.kdeplot(rpl_p, ax=axs[1,0], color='red', clip=bounds)
            axs[1, 0].set_xlabel('Rough Patch Length (pressure) [% Arc Chord Length]')
            axs[1, 0].set_title('Pressure Side RP Length Distribution')
            sns.histplot(rpl_s, ax=axs[1,1], stat='density')
            sns.kdeplot(rpl_s, ax=axs[1,1], color='red', clip=bounds)
            axs[1, 1].set_xlabel('Rough Patch Length (suction) [% Arc Chord Length]')
            axs[1, 1].set_title('Suction Side RP Length Distribution')

            for line in axs:
                for ax in line:
                    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return rh_p, rh_s, rpl_p, rpl_s, lee_labels

    def generate_all_bcs(self, num_samples:int, extra_uncertainty=False, plot=False):
        samples = self.sampler.random_base2(m=int(np.ceil(np.log2(num_samples))))[:num_samples]
        u, rel_u = self.__gen_wind_velocities(samples[:,0], plot=plot)
        turb, ti, tl = self.__gen_turbulence(u, plot=plot)
        shearExp = self.__gen_freestream_shear(u, plot=plot)
        airtemp, rho, mu = self.__gen_air_properties(samples[:,1], plot=plot)
        Re, Ma = self.__gen_dimensionless_nums(rel_u, rho, mu, airtemp, u, extra_uncertainty=extra_uncertainty, plot=plot)
        aoa = self.__gen_angle_of_attack(u, ti, plot=plot)
        rh_p, rh_s, rpl_p, rpl_s, lee_labels = self.__gen_roughness_properties(samples[:,2:4], plot=plot)
        plt.show()

        output_dict = {'u':u, 'rel_u':rel_u, 'turb':turb, 'ti':ti, 'tl':tl, 'shearExp':shearExp, 'airtemp':airtemp,
                       'rho':rho, 'mu':mu, 'nu':mu/rho, 'Re':Re, 'Ma':Ma, 'aoa':aoa, 'rh_p':rh_p, 'rh_s':rh_s,
                       'rpl_p':rpl_p,'rpl_s':rpl_s, 'lee_labels':lee_labels}
        return output_dict


if __name__ == "__main__":
    inflow_settings = {'mean_V':10, 'k_V' : 2.0, 'Iref' : 0.16, 'mean_turb_Lscale': 0.2, 'COV_turb_Lscale': 0.01,
                       'mean_air_temp': 15, 'COV_air_temp': 0.30, 'mean_Dyn_Viscosity' : 1.81397e-5,
                       'COV_Dyn_Viscosity' : 0.01, 'COV_Re':0.02, 'COV_Ma':0.02}
    airfoil_settings = {'blade_span_location': 0.75, 'chord_length':1, 'mean_ang_attack' : 8,
                        'COV_ang_attack' : 0.05, 'min_ang_attack' : -45, 'max_ang_attack' : 45,
                        'mean_rough_patch_length' : 0.0, 'STD_rough_patch_length' : 4.0, 'mean_equ_sand_grain_rough_H' : 0.0,
	                    'STD_sand_grain_rough_H' : 1.5, 'dist_rough_patch' : 'structured' }
    turbine_settings = {'rotor_diameter': 175, 'min_rot_rpm' : 4, 'max_rot_rpm' : 13,
                        'rated_u' : 12, 'cutin_u' : 3, 'cutout_u' : 25, 'height_above_ground' : 100 }
    config_dict = {'inflow_settings': inflow_settings, 'airfoil_settings': airfoil_settings,
                   'turbine_settings': turbine_settings, }
    bc_generator = TurbineCfdBcGenerator(config_dict)
    bc_generator.generate_all_bcs(num_samples=200, plot=True)

