import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config, get_attribute
from Model_equations import *
from crop import CropModel

# Load the weather data
weather_data = pd.read_csv('weather_trondheim.csv')
weather_data['DateTime'] = pd.to_datetime(weather_data['referenceTime']).dt.tz_localize(None)  # Remove timezone info
start_date = pd.to_datetime('2023-07-07')
end_date = pd.to_datetime('2023-07-30')

# Resample weather data to hourly intervals
weather_data.set_index('DateTime', inplace=True)
weather_data_resampled = weather_data.resample('1H').interpolate(method='linear')  # Resample to hourly and interpolate

# Filter weather data between start and end dates
weather_data_filtered = weather_data_resampled[(weather_data_resampled.index >= start_date) & (weather_data_resampled.index <= end_date)]
T_outside_vec = weather_data_filtered['air_temperature'].values
humidity_outside_vec = weather_data_filtered['relative_humidity'].values

# Calculate simulation days dynamically
simulation_days = (end_date - start_date).days  # Include the end date
hours_per_day = 24
N = simulation_days * hours_per_day  # Total number of hours
t_eval = np.arange(0, N * 4)  # Time steps for 4 evaluations per hour, eg every 15 minutes
t_hourly = np.arange(0, N)  # Time vector for hourly data

# Expand hourly data to match t_eval (4 evaluations per hour)
T_outside_vec_expanded = np.repeat(T_outside_vec, 4)
humidity_outside_vec_expanded = np.repeat(humidity_outside_vec, 4)

# Load the spot prices data
spot_prices = pd.read_csv('Spotprices_norway.csv', sep=';')
spot_prices['DateTime'] = pd.to_datetime(spot_prices['Dato/klokkeslett'], format='%Y-%m-%d Kl. %H-%M')
spot_prices_filtered = spot_prices[(spot_prices['DateTime'] >= start_date) & (spot_prices['DateTime'] <= end_date)]
C_price_vec = spot_prices_filtered['NO3'].values.flatten()
C_price_vec_expanded = np.repeat(C_price_vec, 4)  # Repeat hourly spot prices to match t_eval

# Initialize objects and varibles
config = load_config('')
Crop = CropModel(config)

# Constants and parameters
rho_air = get_attribute(config, 'rho_air')  # kg/m^3, density of air
c_air = get_attribute(config, 'c_air')  # J/(kg*C), specific heat capacity of air
lambda_vapor = get_attribute(config, 'lambda')  # J/g, latent heat of vaporization of water
rho_c = get_attribute(config, 'rho_c')  # kg/m^3, density of CO2
c_env = get_attribute(config, 'c_env')
rho_env = get_attribute(config, 'rho_env')
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
U_sup_max = get_attribute(config, 'u_sup_max')  # m^3/s, volume flow rate of supply air
CO2_out = get_attribute(config, 'CO2_out')  # ppm, outside CO2 concentration

# Crop and lighting parameters
A_crop = get_attribute(config, 'A_crop')  # m^2, crop area
CAC = 0.8  # Fraction of crop area cover
c_r = 0.2  # Light reflection coefficient
c_p = get_attribute(config, 'c_p')
eta_light = 0.7  # Lighting efficiency
#f_phot_max = 0.05  # Photosynthetic rate
#PPFD_max = get_attribute(config, 'PPFD_max')

# Evaluated parameters
C_in = rho_air * c_air * V_in
C_env = rho_env * c_env * A_env * x_env

# Input variables from simulated crop model
Q_trans = 30
f_phot = 0 # only for first iteration

# Initialize lists to store the results for plotting
T_in_sim = []
T_env_sim = []
chi_in_sim = []
CO2_in_sim = []
T_sup_vec = []
chi_sup_vec = []

Q_env_vals = []
Q_trans_vals = []
Q_light_vals = []
Q_hvac_vals = []
Q_sum_vals = []

X_ns_vals = [Crop.X_ns]
X_s_vals = [Crop.X_s]
fw_sht_per_plant_vals = [Crop.fresh_weight_shoot_per_plant]
f_phot_vals = []


# Initial conditions for the states: T_in, T_env, chi_in, CO2_in
T_in_current = 20.0
T_env_current = 15.0
chi_in_current = 0.04
CO2_in_current = 1200.0

dt = 1 # the dt has already been compansated for for the climate system, so set to 1

# Control inputs (assuming static control for the simulation)
T_sup = 16.0  # Supply air temperature
#chi_sup = 0.008  # Supply air humidity
CO2_inj = 1  # CO2 injection rate (ppm per second) SJEKK
U_sup = U_sup_max # SJEKK
phi_c_inj = 1 # SJEKK

# Energy assessment initializations
accumulated_energy = 0
Q_humid_vals = []
Q_cool_vals = []
Q_heat_vals = []
total_energy_vals = []
energy_cost_vals = []

for i in range(N * 4):
    # Adjust supply temperature during the simulation for testing the dynamics
    if i > 600:
        T_sup = 22
    if i > 1000:
        T_sup = 13

    # Calculate the next state using the provided ODE functions
    T_in_dt, Q_env_current, Q_trans_current, Q_light_current, Q_hvac_current = Model_equations_T_in_ODE(
        T_in_current, T_env_current, T_sup, U_sup, chi_in_current, Crop, (i // 4) % 24)
    
    chi_sup_val = Model_equations_relative_to_absolute_humidity(humidity_outside_vec_expanded[i], T_sup)

    T_in_next = T_in_current + dt*T_in_dt
    T_env_next = T_env_current + dt*Model_equations_T_env_ODE(T_in_current, T_env_current, T_outside_vec_expanded[i])
    #chi_in_next = chi_in_current + Model_equations_chi_in_ODE(chi_in_current, chi_sup, U_sup, T_in_current, Crop, (i // 4) % 24)
    chi_in_next = chi_in_current + dt*Model_equations_chi_in_ODE(chi_in_current,chi_sup_val, U_sup, T_in_current, Crop, (i // 4) % 24)
    CO2_in_next = CO2_in_current + dt*Model_equations_CO2_in_ODE(CO2_in_current, phi_c_inj, U_sup, CO2_out, f_phot)

    # Append current values to simulation results
    T_in_sim.append(T_in_current)
    T_env_sim.append(T_env_current)
    chi_in_sim.append(chi_in_current)
    CO2_in_sim.append(CO2_in_current)
    T_sup_vec.append(T_sup)
    chi_sup_vec.append(chi_sup_val)

    Q_env_vals.append(Q_env_current)
    Q_trans_vals.append(Q_trans_current)
    Q_light_vals.append(Q_light_current)
    Q_hvac_vals.append(Q_hvac_current)
    Q_sum_vals.append(Q_env_current + Q_trans_current + Q_light_current + Q_hvac_current)

    # Calculations for rotary heat exchanger energy
    T_rot, chi_rot_abs, chi_rot = rotary_heat_exchanger_energy(T_in_current, T_outside_vec_expanded[i], 0.8, chi_in_current, humidity_outside_vec_expanded[i], U_sup)
    
    # Calculations for humidifier energy
    Q_humid_current = humidifier_energy(U_sup, rho_air, lambda_vapor, chi_rot_abs, chi_in_current)
    #print(chi_rot, chi_in_current)

    # Calculations for cooling energy
    Q_cool_current = cooling_coil_energy(U_sup, rho_air, enthalpy(T_rot, chi_rot), enthalpy(T_sup, chi_rot))
    
    # Calculations for heating energy
    Q_heat_current = heating_coil_energy(U_sup, rho_air, c_air, T_sup, T_rot)

    # Sum up all energy for each time step
    total_energy_current = Q_humid_current + Q_cool_current + Q_heat_current
    accumulated_energy += total_energy_current * (15 / 60)  # Convert per 15-minute interval to per hour energy

    # Cost for energy current time step
    energy_cost = total_energy_current * C_price_vec_expanded[i] * 0.25 / 1000 / 100 # converting from hour to 15 minutes, then into kilowatts and lastly from øre to NOK
    energy_cost_vals.append(energy_cost)
    
    # Append the energy values for plotting if necessary
    Q_humid_vals.append(Q_humid_current)
    Q_cool_vals.append(Q_cool_current)
    Q_heat_vals.append(Q_heat_current)
    total_energy_vals.append(total_energy_current)

    # Execute Crop Model simulation step
    X_ns_next, X_s_next, f_phot = Model_equations_Crop_model_simulation_step(Crop, U_sup, (i // 4) % 24, T_in_current, CO2_in_current)
    X_ns_vals.append(X_ns_next)
    X_s_vals.append(X_s_next)
    fw_sht_per_plant_vals.append(Crop.fresh_weight_shoot_per_plant)
    f_phot_vals.append(f_phot)

    # Update the current state variables for the next iteration
    T_in_current = T_in_next
    T_env_current = T_env_next
    chi_in_current = chi_in_next
    CO2_in_current = CO2_in_next

# Convert t_eval to days for x-axis labels
t_days = t_eval / (24 * 4)  # Convert time from hours to days

# Plot Crop states

plt.figure(figsize=(18, 6))

plt.subplot(3, 1, 1)
plt.plot(t_days, X_ns_vals[:len(t_days)], label='X_ns', color='blue')
plt.xlabel('time')
plt.ylabel('X')
plt.title('Crop states')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_days, X_s_vals[:len(t_days)], label='X_s', color='orange')
plt.xlabel('time')
plt.ylabel('X')
plt.title('Crop states')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_days, fw_sht_per_plant_vals[:len(t_days)], label='X_fw,sht', color='green')
plt.xlabel('time')
plt.ylabel('X')
plt.title('Crop states')
plt.legend()
plt.grid(True)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Display the subplots
plt.show()

# Plot f_phot

plt.figure(figsize=(18, 6))  # Increase the width to make the plot wider
plt.plot(t_days, f_phot_vals, label='f_phot', color='purple')
plt.xlabel('Time (days)')
plt.ylabel('Photosynthesis [g m^-2 s^-1]')
plt.title('Photosynthesis')
plt.legend()
plt.grid(True)
plt.show()

# Plot the results for indoor environment
plt.figure(figsize=(18, 6))  # Increase the width to make the plot wider
plt.plot(t_days, T_in_sim, label='Indoor Temperature (°C)', color='blue')
plt.plot(t_days, T_env_sim, label='Envelope Temperature (°C)', color='orange')
plt.plot(t_days, T_sup_vec, label='Supply Air Temperature (°C)', color='green')
plt.axhline(18, color='grey', linestyle='--', label='Lower Bound (18°C)')
plt.axhline(22, color='grey', linestyle='--', label='Upper Bound (22°C)')
plt.xlabel('Time (days)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Simulation Results')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 6))
plt.plot(t_days, chi_in_sim, label='Indoor Humidity (g/m³)')
plt.plot(t_days, chi_sup_vec, label='Supply Air Humidity (g/m³)')
plt.xlabel('Time (days)')
plt.ylabel('Humidity (g/m³)')
plt.title('Humidity Simulation Results')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 6))
plt.plot(t_days, CO2_in_sim, label='Indoor CO₂ Concentration (ppm)')
plt.axhline(CO2_inj, color='r', linestyle='--', label='CO₂ Injection Rate (ppm)')
plt.xlabel('Time (days)')
plt.ylabel('CO₂ Concentration (ppm)')
plt.title('CO₂ Concentration Simulation Results')
plt.legend()
plt.grid(True)
plt.show()

# Create a figure with two subplots: one for outdoor temperature and one for outdoor humidity
plt.figure(figsize=(14, 8))  # Increase the figure size for better visualization

# Subplot 1: Outdoor temperature
plt.subplot(2, 1, 1)
plt.plot(t_hourly / 24, T_outside_vec[:len(t_hourly)], label='Outdoor Temperature (°C)', color='blue')
plt.xlabel('Time (days)')
plt.ylabel('Temperature (°C)')
plt.title('Outdoor Temperature During Simulation Period')
plt.legend()
plt.grid(True)

# Subplot 2: Outdoor humidity
plt.subplot(2, 1, 2)
plt.plot(t_hourly / 24, humidity_outside_vec[:len(t_hourly)], label='Outdoor Humidity (%)', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Humidity (%)')
plt.title('Outdoor Humidity During Simulation Period')
plt.legend()
plt.grid(True)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Display the subplots
plt.show()

# Create a figure with three subplots: one for individual Q values, one for Q_sum_vals, and one for the cumulative Q_sum_vals
plt.figure(figsize=(18, 9))  # Adjusted figure size for wider plots

# Subplot 1: Plot the individual Q values
plt.subplot(3, 1, 1)
plt.plot(t_days, Q_env_vals, label='Q_env (Heat from envelope)', color='blue')
plt.plot(t_days, Q_trans_vals, label='Q_trans (Heat from transpiration)', color='orange')
plt.plot(t_days, Q_light_vals, label='Q_light (Heat from lighting)', color='green')
plt.plot(t_days, Q_hvac_vals, label='Q_hvac (Heat from HVAC)', color='red')
plt.axhline(0, color='black', linewidth=0.9)  # Solid line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Heat Transfer (Q) [Joules]')
plt.title('Heat Transfer Components in ODE1')
plt.legend()
plt.grid(True)

# Subplot 2: Plot the Q_sum_vals
plt.subplot(3, 1, 2)
plt.plot(t_days, Q_sum_vals, label='Q_sum (Total Heat Transfer)', color='purple')
plt.axhline(0, color='black', linewidth=2)  # Thick solid line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Total Heat Transfer (Q) [Joules]')
plt.title('Total Heat Transfer Over Time')
plt.legend()
plt.grid(True)

# Subplot 3: Plot the cumulative Q_sum_vals
Q_sum_cumulative = np.cumsum(Q_sum_vals)  # Calculate the cumulative sum
plt.subplot(3, 1, 3)
plt.plot(t_days, Q_sum_cumulative, label='Cumulative Q_sum', color='purple')
plt.axhline(0, color='black', linewidth=2)  # Thick solid line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Total Heat Transfer (Q) [Joules]')
plt.title('Cumulative Total Heat Transfer Over Time')
plt.legend()
plt.grid(True)

# Display the plots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Print the total energy used over the simulation period
print(f"Total accumulated energy consumption over the simulation period: {accumulated_energy:.2f} W")

# Convert energy values from watts to megawatts (1 MW = 10^6 W)
Q_humid_vals_mw = np.array(Q_humid_vals) / 1e6
Q_cool_vals_mw = np.array(Q_cool_vals) / 1e6
Q_heat_vals_mw = np.array(Q_heat_vals) / 1e6
total_energy_vals_mw = np.array(total_energy_vals) / 1e6
cumulative_energy_use_mw = np.cumsum(total_energy_vals_mw) * (15 / 60)  # Convert and scale cumulative energy

# Convert energy cost values to megawatt-hour costs (assuming energy_cost_vals are in appropriate units)
energy_cost_vals = np.array(energy_cost_vals)  # Convert to MW if needed
cumulative_cost = np.cumsum(energy_cost_vals)  # Convert and scale cumulative cost

"""

# Create a figure with four subplots: energy consumption, cumulative energy use, energy cost, and cumulative cost
plt.figure(figsize=(18, 9))  # Set a larger figure size for better visualization

# Subplot 1: Energy consumption of HVAC components in megawatts
plt.subplot(4, 1, 1)
plt.plot(t_days, Q_humid_vals_mw, label='Humidifier Energy Consumption (MW)', color='blue')
plt.plot(t_days, Q_cool_vals_mw, label='Cooling Coil Energy Consumption (MW)', color='orange')
plt.plot(t_days, Q_heat_vals_mw, label='Heating Coil Energy Consumption (MW)', color='green')
plt.plot(t_days, total_energy_vals_mw, label='Total Energy Consumption (MW)', color='black')
plt.axhline(0, color='black', linewidth=1)  # Reference line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Energy Consumption (MW)')
plt.title('Energy Consumption of HVAC Components')
plt.legend()
plt.grid(True)

# Subplot 2: Cumulative energy use over the simulation period in megawatts
plt.subplot(4, 1, 2)
plt.plot(t_days, cumulative_energy_use_mw, label='Cumulative Total Energy Use (MW)', color='purple')
plt.axhline(0, color='black', linewidth=2)  # Reference line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Energy (MW)')
plt.title('Cumulative Energy Use Over the Simulation Period')
plt.legend()
plt.grid(True)

# Subplot 3: Energy cost per time step
plt.subplot(4, 1, 3)
plt.plot(t_days, 100 * energy_cost_vals, label='Energy Cost per Time Step', color='red')
plt.axhline(0, color='black', linewidth=1)  # Reference line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Energy Cost (Øre)')
plt.title('Energy Cost per Time Step')
plt.legend()
plt.grid(True)

# Subplot 4: Cumulative energy cost over the simulation period
plt.subplot(4, 1, 4)
plt.plot(t_days, cumulative_cost, label='Cumulative Energy Cost', color='purple')
plt.axhline(0, color='black', linewidth=2)  # Reference line at y=0
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Cost (NOK)')
plt.title('Cumulative Energy Cost Over the Simulation Period')
plt.legend()
plt.grid(True)

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Display the subplots
plt.show()

"""