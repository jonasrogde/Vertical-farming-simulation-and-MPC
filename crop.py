import numpy as np
from typing import Dict, Any
from utils import *

class CropModel:
    def __init__(self, config: Dict[str, Any]):
        # Basic configuration parameters
        self.init_FW_per_plant: float = get_attribute(config, 'init_FW_per_plant')
        self.SLA: float = get_attribute(config, 'SLA')
        self.plant_density: float = get_attribute(config, 'plant_density')

        # Physiological parameters
        self.structural_to_nonstructural: float = get_attribute(config, 'structural_to_nonstructural')
        self.dry_weight_fraction: float = get_attribute(config, 'dry_weight_fraction')
        self.leaf_to_shoot_ratio: float = get_attribute(config, 'leaf_to_shoot_ratio')
        self.leaf_diameter: float = get_attribute(config, 'leaf_diameter')

        # Growth coefficients and parameters
        self.c_tau: float = get_attribute(config, 'c_tau')
        self.c_q10_gr: float = get_attribute(config, 'c_q10_gr')
        self.c_gamma: float = get_attribute(config, 'c_gamma')
        self.c_gr_max: float = get_attribute(config, 'c_gr_max')
        self.c_beta: float = get_attribute(config, 'c_beta')
        self.c_a: float = get_attribute(config, 'c_a')
        self.c_resp_sht: float = get_attribute(config, 'c_resp_sht')
        self.c_resp_rt: float = get_attribute(config, 'c_resp_rt')
        self.c_q10_resp: float = get_attribute(config, 'c_q10_resp')
        self.c_K: float = get_attribute(config, 'c_K')
        self.c_lar: float = get_attribute(config, 'c_lar')
        self.c_w: float = get_attribute(config, 'c_w')
        self.c_Gamma: float = get_attribute(config, 'c_Gamma')
        self.c_q10_Gamma: float = get_attribute(config, 'c_q10_Gamma')
        self.c_car_1: float = get_attribute(config, 'c_car_1')
        self.c_car_2: float = get_attribute(config, 'c_car_2')
        self.c_car_3: float = get_attribute(config, 'c_car_3')
        self.c_epsilon: float = get_attribute(config, 'c_epsilon')
        self.c_p: float = get_attribute(config, 'c_p')
        self.g_stm: float = 0.005
        self.g_bnd: float = 0.007

        # Initialize dynamic attributes
        self.DAT: int = 0  # Days After Transplanting
        self.set_dynamic_attributes()

        # Initial state
        self.init_state = np.array([self.X_ns, self.X_s])

    def set_climate_model(self, climate_model):
        self.climate_model = climate_model

    def set_dynamic_attributes(self):
        self.fresh_weight_shoot_per_plant: float = self.init_FW_per_plant
        self.fresh_weight_shoot: float = self.fresh_weight_shoot_per_plant * self.plant_density
        self.dry_weight: float = self.fresh_weight_shoot * self.dry_weight_fraction / (1 - self.c_tau)
        self.dry_weight_per_plant: float = self.dry_weight / self.plant_density
        self.structural_dry_weight_per_plant: float = self.dry_weight_per_plant * self.structural_to_nonstructural
        self.X_ns: float = self.dry_weight * (1 - self.structural_to_nonstructural)
        self.X_s: float = self.dry_weight * self.structural_to_nonstructural
        # self.LAI: float = SLA_to_LAI(SLA=self.SLA, c_tau=self.c_tau, leaf_to_shoot_ratio=self.leaf_to_shoot_ratio, X_s=self.X_s, X_ns=self.X_ns)
        self.LAI: float = biomass_to_LAI(self.X_s, self.c_lar, self.c_tau)
        self.CAC: float = LAI_to_CAC(self.LAI)
        self.f_phot: float = 0

    def set_fresh_weight_shoot(self):
        self.fresh_weight_shoot = self.dry_weight * (1 - self.c_tau) / self.dry_weight_fraction
        self.fresh_weight_shoot_per_plant = self.fresh_weight_shoot / self.plant_density

    def update_values(self, X_ns: float, X_s: float):
        self.X_ns = X_ns
        self.X_s = X_s
        self.dry_weight = X_ns + X_s
        self.dry_weight_per_plant = self.dry_weight / self.plant_density
        self.structural_dry_weight_per_plant = self.X_s / self.plant_density
        self.set_fresh_weight_shoot()
        # self.LAI = SLA_to_LAI(SLA=self.SLA, c_tau=self.c_tau, leaf_to_shoot_ratio=self.leaf_to_shoot_ratio, X_s=self.X_s, X_ns=self.X_ns)
        self.LAI: float = biomass_to_LAI(self.X_s, self.c_lar, self.c_tau)
        self.CAC = LAI_to_CAC(self.LAI)

    def print_attributes(self, *args):
        if args:
            for attr_name in args:
                print(f"{attr_name}: {getattr(self, attr_name, 'Attribute not found')}")
        else:
            for attr, value in vars(self).items():
                print(f"{attr}: {value}")

    def return_photosynthesis(self, CO2_in, T_in, g_bnd, g_stm, U_par, fun_type='rectangular'):

        #CO2_in = 1200
        #T_in = 24

        #fun_type='exponential'

        CO2_ppm = CO2_in
        g_car = self.c_car_1 * T_in**2 + self.c_car_2 * T_in + self.c_car_3
        g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
        Gamma = self.c_Gamma * self.c_q10_Gamma ** ((T_in - 20) / 10)
        epsilon_biomass = self.c_epsilon * (CO2_ppm - Gamma) / (CO2_ppm + 2 * Gamma)
        if fun_type=='exponential':
            A_sat = g_CO2 * self.c_w * (CO2_ppm - Gamma)
            k_slope =epsilon_biomass / A_sat
            f_phot_max = A_sat * (1 - np.exp(-k_slope * U_par))
        elif fun_type=='rectangular':
            f_phot_max = (epsilon_biomass * U_par * g_CO2 * self.c_w * (CO2_ppm - Gamma)) / (epsilon_biomass * U_par + g_CO2 * self.c_w * (CO2_ppm - Gamma))
        return f_phot_max
    
    def biomass_ode(self, X_ns: float, X_s: float, T_in: float, CO2_in: float, U_par: float, PPFD: float, g_bnd: float, g_stm: float):

        #CO2_in = 1200
        #T_in = 24


        CO2_ppm = CO2_in
        g_car = self.c_car_1 * T_in**2 + self.c_car_2 * T_in + self.c_car_3
        g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
        Gamma = self.c_Gamma * self.c_q10_Gamma ** ((T_in - 20) / 10)
        epsilon_biomass = self.c_epsilon * (CO2_ppm - Gamma) / (CO2_ppm + 2 * Gamma)
        
        f_phot_max = (epsilon_biomass * U_par * g_CO2 * self.c_w * (CO2_ppm - Gamma)) / (epsilon_biomass * U_par + g_CO2 * self.c_w * (CO2_ppm - Gamma))
        f_phot = (1 - np.exp(-self.c_K * self.LAI)) * f_phot_max
        self.f_phot = f_phot
        self.p_phot_max = f_phot_max
        f_resp = (self.c_resp_sht * (1 - self.c_tau) * X_s + self.c_resp_rt * self.c_tau * X_s) * self.c_q10_resp ** ((T_in - 25) / 10)
        
        self.f_resp = f_resp
        r_gr = self.c_gr_max * X_ns / (self.c_gamma * X_s + X_ns) * self.c_q10_gr ** ((T_in - 20) / 10)

        # For testing purposes
        #r_gr = 1e-6 * self.c_q10_gr ** ((T_in - 20) / 10)
        dX_ns = self.c_a * f_phot - r_gr * X_s - f_resp - (1 - self.c_beta) / self.c_beta * r_gr * X_s
        dX_s = r_gr * X_s

        return dX_ns, dX_s

    def combined_ODE(self, state, control_inputs, data):
        T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_ns, X_s = state
        self.update_values(X_ns, X_s)

        PPFD = control_inputs[6]
        U_par = PPFD * self.c_p
        
        g_bnd = 1 / aerodynamical_resistance_eq(uninh_air_vel=self.climate_model.air_vel, LAI=self.LAI, leaf_diameter=self.leaf_diameter)
        g_stm = 1 / stomatal_resistance_eq(PPFD=PPFD)
        
        dNS_dt, dS_dt = self.biomass_ode(X_ns=X_ns, X_s=X_s, T_in=T_in, CO2_in=CO2_in, U_par=U_par, PPFD=PPFD, g_bnd=g_bnd, g_stm=g_stm)
        return np.array([dNS_dt, dS_dt])
