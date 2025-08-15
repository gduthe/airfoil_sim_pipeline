import numpy as np
from scipy.stats import qmc, weibull_min, norm, cauchy, halfnorm
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import seaborn as sns

class AirfoilCfdBcGenerator:
    def __init__(self, config_dict:dict):
        """
        Space filling boundary condition generator for airfoils
        """

        self.inflow_settings = config_dict['inflow_settings']
        self.airfoil_settings = config_dict['airfoil_settings']

        self.sampler = qmc.Sobol(d=7, scramble=True)

        # plotting settings
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2


    def __gen_inflow_velocity(self, sobol_samples:np.array, plot=False):
        # calculate the probabilities corresponding to the bounds
        bounds = np.array([self.inflow_settings['min_vel'], self.inflow_settings['max_vel']])

        # scale and shift samples to cover bounds
        u = bounds[0] + sobol_samples*(bounds[1]-bounds[0])

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
            sns.histplot(u, ax=ax, stat='density')
            sns.kdeplot(u, ax=ax, color='red', clip=bounds)
            ax.set_xlabel('Inflow Velocity [m/s]')
            ax.set_title('Inflow Velocity Distribution')
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return u

    def __gen_turbulence(self, sobol_samples:np.array, u:np.array, plot=False):
        bounds = np.array([self.inflow_settings['min_turb_i'], self.inflow_settings['max_turb_i']])

        # scale and shift samples to cover bounds
        ti = bounds[0] + sobol_samples[:, 0] * (bounds[1] - bounds[0])

        bounds = np.array([self.inflow_settings['min_turb_l'], self.inflow_settings['max_turb_l']])

        # scale and shift samples to cover bounds
        tl = bounds[0] + sobol_samples[:,1] * (bounds[1] - bounds[0])


        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
            sns.scatterplot(x=u, y=ti, ax=axs[0])
            axs[0].set_xlabel('Inflow Velocity [m/s]')
            axs[0].set_ylabel('Turbulence Intensity [%]')
            axs[0].set_title('Turbulence Intensity vs. Inflow Velocity')
            sns.scatterplot(x=u, y=tl*100, ax=axs[1])
            axs[1].set_xlabel('Inflow Velocity [m/s]')
            axs[1].set_ylabel('Turbulence Length [% Chord L]')
            axs[1].set_title('Turbulence Length vs. Inflow Velocity')
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return ti, tl


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

    def __gen_dimensionless_nums(self, u:np.array, rho:np.array, mu:np.array, airtemp:np.array ,extra_uncertainty=False,
                                 plot=False):
        mean_Re = rho* u* self.airfoil_settings['chord_length']/ mu

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
        mean_Ma = u/np.sqrt(specific_heat_ratio*adiabatic_gas_const*airtemp / mol_mass_gas) # assuming ideal gas

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
            axs[0].set_xlabel('Inflow Velocity [m/s]')
            axs[0].set_ylabel('Reynolds Number [-]')
            axs[0].set_title('Reynolds Number vs. Inflow Velocity')
            sns.scatterplot(x=u, y=Ma, ax=axs[1])
            axs[1].set_xlabel('Inflow Velocity [m/s]')
            axs[1].set_ylabel('Mach Number [-]')
            axs[1].set_title('Mach Number vs. Inflow Velocity')
            for ax in axs:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return Re, Ma

    def __gen_angle_of_attack(self, sobol_samples:np.array, u:np.array, plot=False):
        # get the bounds of the sobol sampling
        bounds = np.array([self.airfoil_settings['min_ang_attack'], self.airfoil_settings['max_ang_attack']])

        # scale and shift samples to cover bounds
        aoa = bounds[0] + sobol_samples*(bounds[1]-bounds[0])

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 6))
            sns.scatterplot(x=u, y=aoa, ax=ax)
            ax.set_xlabel('Inflow Velocity [m/s]')
            ax.set_ylabel('Angle of Attack [°]')
            ax.set_title('Angle of Attack vs. Inflow Velocity')
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            fig.tight_layout(pad=3.0)

        return aoa

    def __gen_roughness_properties(self, sobol_samples:np.array, plot=False):
        # grain roughness size is 1.3 times larger on pressure compared to suction side
        roughness_height_ratio = 1.3
        qL = 80 # lower bound of the roughness size
        qH = 350 # upper bound of the roughness size
        rh_bounds = np.array([qL, qH])
        mu = self.airfoil_settings['mean_rough_H']
        sigma = self.airfoil_settings['STD_rough_H']

        # calculate the probabilities corresponding to the bounds, then scale and shift random samples
        P_bounds = halfnorm.cdf(rh_bounds, loc=mu, scale=sigma)
        x = P_bounds[0] + sobol_samples[:,0] * (P_bounds[1] - P_bounds[0])

        # equivalent sand grain roughness heights
        rh_p = halfnorm.ppf(x, loc=mu, scale=sigma)
        rh_s = halfnorm.ppf(x, loc=mu, scale=sigma) / roughness_height_ratio

        # patch length is 1.3 times larger on pressure compared to suction side
        patch_length_ratio = 1.3

        lL = 0 # lower bound of the Roughness patch length, percent of chord length
        lH = 15 # upper bound of the Roughness patch length on the pressure side, percent of chord length
        rl_bounds = np.array([lL, lH])
        mu = self.airfoil_settings['mean_rough_patch_length']
        sigma = self.airfoil_settings['STD_rough_patch_length']

        # calculate the probabilities corresponding to the bounds, then scale and shift random samples
        P_bounds = halfnorm.cdf(rl_bounds, loc=mu, scale=sigma)
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
            sns.kdeplot(rh_p, ax=axs[0,0], color='red', clip=rh_bounds*roughness_height_ratio)
            axs[0,0].set_xlabel('Roughness Height (pressure side) [umm]')
            axs[0,0].set_title('Pressure Side Roughness Height Distribution')
            sns.histplot(rh_s, ax=axs[0,1], stat='density')
            sns.kdeplot(rh_s, ax=axs[0,1], color='red', clip=rh_bounds)
            axs[0, 1].set_xlabel('Roughness Height (suction side) [umm]')
            axs[0, 1].set_title('Suction Side Roughness Height Distribution')
            sns.histplot(rpl_p, ax=axs[1,0], stat='density')
            sns.kdeplot(rpl_p, ax=axs[1,0], color='red', clip=rl_bounds)
            axs[1, 0].set_xlabel('Rough Patch Length (pressure) [% Arc Chord Length]')
            axs[1, 0].set_title('Pressure Side RP Length Distribution')
            sns.histplot(rpl_s, ax=axs[1,1], stat='density')
            sns.kdeplot(rpl_s, ax=axs[1,1], color='red', clip=rl_bounds)
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
        u = self.__gen_inflow_velocity(samples[:,0], plot=plot)
        ti, tl = self.__gen_turbulence(samples[:,1:3], u, plot=plot)
        airtemp, rho, mu = self.__gen_air_properties(samples[:,3], plot=plot)
        Re, Ma = self.__gen_dimensionless_nums(u, rho, mu, airtemp, extra_uncertainty=extra_uncertainty, plot=plot)
        aoa = self.__gen_angle_of_attack(samples[:,4], u, plot=plot)
        rh_p, rh_s, rpl_p, rpl_s, lee_labels = self.__gen_roughness_properties(samples[:,5:7], plot=plot)
        plt.show()

        output_dict = {'u':u, 'ti':ti, 'tl':tl, 'airtemp':airtemp,
                       'rho':rho, 'mu':mu, 'nu':mu/rho, 'Re':Re, 'Ma':Ma, 'aoa':aoa, 'rh_p':rh_p, 'rh_s':rh_s,
                       'rpl_p':rpl_p,'rpl_s':rpl_s, 'lee_labels':lee_labels}
        return output_dict


if __name__ == "__main__":
    inflow_settings = {'min_vel':1, 'max_vel': 100, 'k_V' : 2.0,  'min_turb_i': 0, 'max_turb_i': 20,
                       'min_turb_l': 0.01, 'max_turb_l': 0.5, 'mean_air_temp': 15, 'COV_air_temp': 0.30,
                       'mean_Dyn_Viscosity' : 1.81397e-5,'COV_Dyn_Viscosity' : 0.01, 'COV_Re':0.02, 'COV_Ma':0.02}
    airfoil_settings = {'chord_length':1,'min_ang_attack' : -45, 'max_ang_attack' : 45,
                        'mean_rough_patch_length' : 0.0, 'STD_rough_patch_length' : 5.0, 'mean_rough_H' : 0.0,
	                    'STD_rough_H' : 100, 'dist_rough_patch' : 'structured' }
    config_dict = {'inflow_settings': inflow_settings, 'airfoil_settings': airfoil_settings}
    bc_generator = AirfoilCfdBcGenerator(config_dict)
    bc_generator.generate_all_bcs(num_samples=200, plot=True)

