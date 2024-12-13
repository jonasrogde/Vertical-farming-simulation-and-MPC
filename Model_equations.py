import pandas as pd
import casadi as ca
import numpy as np
import datetime
import matplotlib.pyplot as plt
from utils import *

dt = 900 # 900 seconds in 15 minutes which is the duration of every time step, dt is used for the evaluation of change per time step in the plant model

config = load_config('')
# get_attribute(config, '')

# Constants and parameters
rho_air = get_attribute(config, 'rho_air')  # kg/m^3, density of air
c_air = get_attribute(config, 'c_air')  # J/(kg*C), specific heat capacity of air
lambda_vapor = get_attribute(config, 'lambda')  # J/g, latent heat of vaporization of water
rho_c = get_attribute(config, 'rho_c')  # kg/m^3, density of CO2
p_atm = get_attribute(config, "p")
c_water = get_attribute(config, "c_water")

# Area and volume related parameters
V_in = get_attribute(config, 'V_in')  # m^3, indoor volume
A_env = get_attribute(config, 'A_env')  # m^2, area of building envelope
x_env = get_attribute(config, 'x_env')  # m, thickness of the building envelope

# Heat transfer coefficients
alpha_env = get_attribute(config, 'alpha_env')  # W/(m^2*C), heat transfer coefficient for building envelope
alpha_ext = get_attribute(config, 'alpha_ext')  # W/(m^2*C), heat transfer coefficient for external environment

# HVAC parameters
U_sup_max = get_attribute(config, 'u_sup_max') # m^3/s, volume flow rate of supply air (example value)
#T_sup = get_attribute(config, '')  # C, supply air temperature (example value)
#chi_sup = 0.01  # g/m^3, absolute humidity of supply air (example value)
CO2_out = get_attribute(config, 'CO2_out')  # ppm, outside CO2 concentration (example value)

# Crop and lighting parameters
A_crop = get_attribute(config, 'A_crop')  # m^2, crop area (example value)
# U_par = 200  # W/m^2, PAR irradiance (example value)
#CAC = 0.8  # fraction of crop area cover (example value) #SJEKK
c_r = 0.2  # light reflection coefficient
c_p = get_attribute(config, 'c_p')
eta_light = 0.7  # lighting efficiency
f_phot_max = 0.05  # photosynthetic rate (example value)
PPFD_max = get_attribute(config, 'PPFD_max')

# Evaluated parameters
C_in = rho_air * c_air * V_in

# Input variables from simulated crop model
#Q_trans_ = 30
#f_phot_ = f_phot_max * CAC

# Light schedule pattern
start_date = '2023-07-10'
end_date = '2023-07-30'
start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
num_days = (end_datetime - start_datetime).days
base_pattern = np.concatenate([np.full(18, PPFD_max), np.full(6, 0)])
PPFD = np.tile(base_pattern, num_days)

# ------ PLANT MODEL EQUATIONS -------

# STATE VARIABLES : X_ns, X_s
# SJEKK hva er forskjellen på saturation_curve_type lik rectangular og exponential?

def Model_equations_Crop_model_simulation_step(Crop, U_sup, hour, T_in, CO2_in):

    #T_in = 24
    #CO2_in = 1200

    u_light = PPFD[hour] # SJEKK om dette tilsvarer u_light 
    U_par = c_p * u_light # SJEKK om dette tilsvarer PAR_flux
    LAI = biomass_to_LAI(Crop.X_s, Crop.c_lar, Crop.c_tau)
    g_stm = 1 / stomatal_resistance_eq(u_light)
    g_bnd = 1 / aerodynamical_resistance_eq(U_sup, LAI=LAI, leaf_diameter=Crop.leaf_diameter)

    g_car = Crop.c_car_1 * T_in**2 + Crop.c_car_2 * T_in + Crop.c_car_3
    g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
    Gamma = Crop.c_Gamma * Crop.c_q10_Gamma ** ((T_in - 20) / 10)

    epsilon_biomass = Crop.c_epsilon * (CO2_in - Gamma) / (CO2_in + 2 * Gamma)
    #f_phot_max = (epsilon_biomass * U_par * g_CO2 * Crop.c_w * (CO2_in - Gamma)) / (epsilon_biomass * U_par + g_CO2 * Crop.c_w * (CO2_in - Gamma))
    f_phot_max = Crop.return_photosynthesis(CO2_in, T_in, g_bnd, g_stm, U_par)

    #print("direct: ", f_phot_max)
    #print("function: ", Crop.return_photosynthesis(CO2_in, T_in, g_bnd, g_stm, U_par))

    f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max

    f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max 
    f_resp = (Crop.c_resp_sht * (1 - Crop.c_tau) * Crop.X_s + Crop.c_resp_rt * Crop.c_tau * Crop.X_s) * Crop.c_q10_resp ** ((T_in - 25) / 10)
    dw_to_fw = (1 - Crop.c_tau) / (Crop.dry_weight_fraction * Crop.plant_density)
    r_gr = Crop.c_gr_max * Crop.X_ns / (Crop.c_gamma * Crop.X_s + Crop.X_ns + 0.001) * Crop.c_q10_gr ** ((T_in - 20) / 10)
    dX_ns = Crop.c_a * f_phot - r_gr * Crop.X_s - f_resp - (1 - Crop.c_beta) / Crop.c_beta * r_gr * Crop.X_s
    dX_s = r_gr * Crop.X_s

    # Update variables
    Crop.update_values(Crop.X_ns + dt * dX_ns, Crop.X_s + dt * dX_s)
    Crop.g_stm = g_stm
    Crop.g_bnd = g_bnd

    return Crop.X_ns, Crop.X_s, f_phot


# ------ CLIMATE MODEL EQUATIONS -------

def Model_equations_T_in_ODE(T_in, T_env, T_sup, U_sup, chi_in, Crop, hour):
    C_in = rho_air * c_air * V_in
    Q_env = alpha_env * A_env * (T_env - T_in)

    U_par = c_p * PPFD[hour]    #hour is the time step (since every time step is one hour), so at 12 pm the first day hour=0, and at 6 am at day 2 hour= 24 + 6=30
    P_light = (U_par / eta_light) * A_crop
    Q_light_ineff = (1 - eta_light) * P_light * A_crop #SJEKK
    Q_light_refl = eta_light * P_light * c_r * A_crop
    Q_light = 0.15*(Q_light_ineff + Q_light_refl)

    e_s = 1
    T_crop = T_in
    chi_crop = chi_in + (rho_air * c_air) / lambda_vapor * e_s * (T_crop - T_in)

    R_net = U_par * Crop.CAC * (1 - c_r) * A_crop
    Q_lat_plant = Crop.CAC * lambda_vapor * (chi_crop - chi_in) / (Crop.g_stm - Crop.g_bnd) * A_crop

    Q_trans = R_net - Q_lat_plant

    Q_hvac = U_sup * rho_air * c_air * (T_sup - T_in)

    dT_in_dt = (1/C_in) * (Q_env + Q_trans + Q_light + Q_hvac)
    return dT_in_dt, Q_env, Q_trans, Q_light, Q_hvac

def Model_equations_T_env_ODE(T_in, T_env, T_out):
    C_env = rho_air * c_air * A_env * x_env
    Q_env_in = alpha_env * A_env * (T_in - T_env)
    Q_env_out = alpha_ext * A_env * (T_out - T_env)

    dT_env_dt = (1/C_env) * (Q_env_in + Q_env_out)
    return dT_env_dt

def Model_equations_chi_in_ODE(chi_in, chi_sup, U_sup, T_in, Crop, hour):
    U_par = c_p * PPFD[hour] # hour is the hour of the day. it is used to determine whether the light are on or off
    e_s = 1
    T_crop = T_in
    chi_crop = chi_in + (rho_air * c_air) / lambda_vapor * e_s * (T_crop - T_in)

    R_net = U_par * Crop.CAC * (1 - c_r) * A_crop
    Q_lat_plant = Crop.CAC * lambda_vapor * (chi_crop - chi_in) / (Crop.g_stm - Crop.g_bnd) * A_crop

    Q_trans = R_net - Q_lat_plant

    phi_trans = Q_trans / lambda_vapor
    phi_hvac = U_sup * (chi_sup - chi_in)

    dchi_in_dt = (1/V_in) * (phi_trans + phi_hvac)
    return dchi_in_dt

def Model_equations_CO2_in_ODE(CO2_in, phi_c_inj, U_sup, CO2_out, f_phot):
    phi_c_ass = f_phot * A_crop

    phi_c_hvac = U_sup * (rho_c / 1000) * (CO2_in - CO2_out)

    dCO2_in_dt = (1/(V_in * rho_c)) * (phi_c_inj - phi_c_ass + phi_c_hvac)
    return dCO2_in_dt

def Model_equations_relative_to_absolute_humidity(RH, T):
    # Calculate the saturation vapor pressure (in hPa)
    e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    # Calculate absolute humidity (in g/m^3)
    AH = (e_s * RH * 2.1674) / (T + 273.15)
    return AH

def rotary_heat_exchanger_energy(T_in, T_out, U_rot, chi_in, chi_out, U_sup):
    # Ensure positive recovery factors by limiting them between 0 and 1
    eta_T = max(0, min(1, U_rot * (T_in - T_out) / (T_in - T_out) if T_in != T_out else 0))
    eta_chi = max(0, min(1, U_rot * (chi_in - chi_out) / (chi_in - chi_out) if chi_in != chi_out else 0))

    # Calculate the outgoing air conditions
    T_rot = U_rot * eta_T * (T_in - T_out) + T_out
    chi_rot = U_rot * eta_chi * (chi_in - chi_out) + chi_out

    # Ensure T_rot and chi_rot remain within realistic ranges
    T_rot = max(min(T_rot, T_in), T_out)
    chi_rot = max(min(chi_rot, chi_in), chi_out)

    chi_rot_abs = Model_equations_relative_to_absolute_humidity(chi_rot, T_rot)

    return T_rot, chi_rot_abs, chi_rot

def humidifier_energy(U_sup, rho_air, lambda_vapor, chi_humid, chi_cool, eta_humid=1.0):
    # Calculate thermal energy required for the humidifier, ensuring positive values
    U_humid = max(0, U_sup * rho_air * lambda_vapor * (chi_humid - chi_cool))
    Q_humid = eta_humid * U_humid
    return Q_humid

def cooling_coil_energy(U_sup, rho_air, h_fan, h_cool):
    # Ensure positive energy consumption for cooling
    U_cool = max(0, U_sup * rho_air * (h_fan - h_cool))
    return U_cool

def heating_coil_energy(U_sup, rho_air, c_air, T_heat, T_cool):
    # Calculate heating energy and ensure positive values
    U_heat = max(0, U_sup * rho_air * c_air * (T_heat - T_cool))
    return U_heat

def saturation_vapor_pressure(T):
    """Calculate the saturation vapor pressure in Pa for a given temperature in Celsius."""
    return 610.78 * np.exp((17.27 * T) / (T + 237.3))

def humidity_ratio(relative_humidity, T):
    """Calculate the humidity ratio (w) for a given relative humidity and temperature in Celsius."""
    p_sat = saturation_vapor_pressure(T)
    return 0.622 * (relative_humidity * p_sat) / (p_atm - (relative_humidity * p_sat))

def enthalpy(T, relative_humidity):
    """Calculate the enthalpy of moist air in J/kg."""
    w = humidity_ratio(relative_humidity, T)
    return c_air * T + w * (c_water * T + lambda_vapor)

def print_figs():
    print("T_in: ", T_in_list)
    print("T_env: ", T_env_list)
    print("chi_in: ", chi_in_list)
    print("CO2_in: ", CO2_in_list)

    # Plot T_in
    plt.figure(figsize=(8, 5))
    plt.plot(T_in_list, marker='o', linestyle='-', color='blue')
    plt.title('T_in Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('T_in (°C)')
    plt.show()

    # Plot T_env
    plt.figure(figsize=(8, 5))
    plt.plot(T_env_list, marker='o', linestyle='-', color='green')
    plt.title('T_env Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('T_env (°C)')
    plt.show()

    # Plot chi_in
    plt.figure(figsize=(8, 5))
    plt.plot(chi_in_list, marker='o', linestyle='-', color='purple')
    plt.title('chi_in Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('chi_in')
    plt.show()

    # Plot CO2_in
    plt.figure(figsize=(8, 5))
    plt.plot(CO2_in_list, marker='o', linestyle='-', color='red')
    plt.title('CO2_in Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('CO2_in (ppm)')
    plt.show()