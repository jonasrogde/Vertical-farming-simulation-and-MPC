import pandas as pd
import casadi as ca
import numpy as np
import datetime
import matplotlib.pyplot as plt
from Model_equations import T_in_ODE, T_env_ODE, chi_in_ODE, CO2_in_ODE

# Constants and parameters
dt = 1  # 1 hour time step
N = 24  # 24-hour prediction horizon

# Control inputs
# [T_sup, Chi_sup, phi_c_inj, U_sup]

# Load the weather data
weather_data = pd.read_csv('weather_trondheim.csv')
weather_data['DateTime'] = pd.to_datetime(weather_data['referenceTime'])
start_date = '2023-07-10'
end_date = '2023-07-30'
weather_data_filtered = weather_data[(weather_data['DateTime'] >= start_date) & (weather_data['DateTime'] <= end_date)]
T_outside_vec = weather_data_filtered['air_temperature'].values
humidity_outside_vec = weather_data_filtered['relative_humidity'].values  # Assuming this column exists

# Find number of days
start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
num_days = (end_datetime - start_datetime).days

# Load the spot prices data
spot_prices = pd.read_csv('Spotprices_norway.csv', sep=';')
spot_prices['DateTime'] = pd.to_datetime(spot_prices['Dato/klokkeslett'], format='%Y-%m-%d Kl. %H-%M')
spot_prices_filtered = spot_prices[(spot_prices['DateTime'] >= start_date) & (spot_prices['DateTime'] <= end_date)]
C_price_vec = spot_prices_filtered['NO3'].values.flatten()

# Initialize the state variables
T_in = 22.0  # Initial indoor temperature (°C)
T_env = 20.0  # Initial environment temperature (°C)
chi_in = 0.01  # Initial absolute humidity (g/m^3)
CO2_in = 400  # Initial CO2 concentration (ppm)

# Initialize variables for the MPC loop
U0 = [20, 0.005, 0, 20]  # Initial guess for control inputs, four input #SJEKK
T_current_val = T_in  # Current indoor temperature
T_env_current_val = T_env  # Current environment temperature
chi_in_current_val = chi_in  # Current absolute humidity
CO2_in_current_val = CO2_in  # Current CO2 concentration

results = []
cumulative_cost = 0
cumulative_costs = []



for i in range(len(T_outside_vec) - N):
    # Set up the control input for the optimization
    U = ca.SX.sym('U', N)  # Control inputs over the horizon
    
    # Initialize the total cost function
    total_cost = 0
    
    # Reset the state variables for this iteration
    T_current = T_current_val
    T_env_current = T_env_current_val
    chi_in_current = chi_in_current_val
    CO2_in_current = CO2_in_current_val
    
    for k in range(N):
        # Update state variables using the ODEs
        dT_in = T_in_ODE(T_current, T_env_current, U[k], U[k], U[k])
        dT_env = T_env_ODE(T_current, T_env_current, T_outside_vec[i + k])
        dchi_in = chi_in_ODE(U[k], chi_in_current)
        dCO2_in = CO2_in_ODE(CO2_in_current, U[k], U[k])
        
        T_current = T_current + dT_in * dt
        T_env_current = T_env_current + dT_env * dt
        chi_in_current = chi_in_current + dchi_in * dt
        CO2_in_current = CO2_in_current + dCO2_in * dt
        
        # Calculate the cost components
        penalty = 10000  # Penalty factor for extreme deviations
        T_low = 18
        T_high = 22
        penalty_cost = penalty * (ca.fmax(0, T_low - T_current) ** 2 + ca.fmax(0, T_current - T_high) ** 2)
        energy_cost = U[k] ** 2 * C_price_vec[i + k]
        
        total_cost += energy_cost + penalty_cost
    
    # Define the NLP problem
    nlp = {
        'x': U,  # Control inputs
        'f': total_cost  # Objective function
    }
    
    # Solve the NLP problem using CasADi
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=U0)
    u_opt = sol['x']
    
    # Apply the first control input
    hourly_cost = float(C_price_vec[i] * (u_opt[0] ** 2))
    cumulative_cost += hourly_cost
    
    # Convert CasADi SX variables to float
    T_current_numeric = float(T_current)
    T_env_current_numeric = float(T_env_current)
    chi_in_current_numeric = float(chi_in_current)
    CO2_in_current_numeric = float(CO2_in_current)
    
    results.append({
        'time': weather_data_filtered['DateTime'].iloc[i],
        'T_inside': T_current_numeric,
        'T_env': T_env_current_numeric,
        'chi_in': chi_in_current_numeric,
        'CO2_in': CO2_in_current_numeric,
        'u_opt': float(u_opt[0]),
        'hourly_cost': hourly_cost
    })
    cumulative_costs.append(cumulative_cost)
    
    # Update the initial guess for the next optimization
    U0 = np.append(u_opt[1:], u_opt[-1])

# Convert results to a DataFrame for easier analysis and plotting
results_df = pd.DataFrame(results)
results_df['cumulative_cost'] = cumulative_costs

# Check the length of results_df
print(f"Number of time steps: {len(results_df)}")

# Plot the weather conditions (temperature and humidity)
plt.figure(figsize=(14, 7))
plt.plot(weather_data_filtered['DateTime'], T_outside_vec, label='Outdoor Temperature (°C)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Outdoor Temperature During Simulation Period')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(weather_data_filtered['DateTime'], humidity_outside_vec, label='Outdoor Humidity (%)')
plt.xlabel('Time')
plt.ylabel('Relative Humidity (%)')
plt.title('Outdoor Humidity During Simulation Period')
plt.legend()
plt.grid(True)
plt.show()

# Plot the electricity spot prices
plt.figure(figsize=(14, 7))
plt.plot(spot_prices_filtered['DateTime'], C_price_vec, label='Electricity Spot Price (NOK per kWh)')
plt.xlabel('Time')
plt.ylabel('Electricity Spot Price (NOK per kWh)')
plt.title('Electricity Spot Prices During Simulation Period')
plt.legend()
plt.grid(True)
plt.show()

# Plot the MPC-controlled results
plt.figure(figsize=(14, 7))
plt.plot(results_df['time'], results_df['T_inside'], label='Indoor Temperature (°C)')
plt.axhline(y=20, color='r', linestyle='--', label='Target Temperature (20°C)')
plt.fill_between(results_df['time'], 18, 22, color='green', alpha=0.3, label='Acceptable Range (18-22°C)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('MPC-Controlled Indoor Temperature')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(results_df['time'], results_df['chi_in'], label='Absolute Humidity (g/m^3)')
plt.xlabel('Time')
plt.ylabel('Absolute Humidity (g/m^3)')
plt.title('MPC-Controlled Absolute Humidity')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(results_df['time'], results_df['CO2_in'], label='CO2 Concentration (ppm)')
plt.xlabel('Time')
plt.ylabel('CO2 Concentration (ppm)')
plt.title('MPC-Controlled CO2 Concentration')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(results_df['time'], results_df['cumulative_cost'], label='Cumulative Cost (NOK)')
plt.xlabel('Time')
plt.ylabel('Cumulative Cost (NOK)')
plt.title('Cumulative Electricity Cost Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Check the final cumulative cost
print(f"Final Cumulative Cost: {cumulative_cost:.2f} NOK")