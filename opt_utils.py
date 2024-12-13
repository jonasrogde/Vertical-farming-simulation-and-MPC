import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import casadi as ca
import json

def load_opt_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

# Load climate model
opt_config = load_opt_config('opt_config_casadi.json')

def get_params(opt_config):
    # Integrator settings
    N_horizon = opt_config['N_horizon']
    ts  = opt_config['ts']
    
    solver = opt_config['solver']

    # Growth settings
    photoperiod_length = opt_config['photoperiod_length']
    u_max = opt_config['u_max']
    u_min = opt_config['u_min']
    min_DLI = opt_config['min_DLI']
    max_DLI = opt_config['max_DLI']
    l_end_mass = opt_config['lower_terminal_mass']
    u_end_mass = opt_config['upper_terminal_mass']

    is_rate_degradation = opt_config['is_rate_degradation']
    c_degr = opt_config['c_degr']
    return N_horizon, ts, solver, photoperiod_length, u_max, u_min, min_DLI, max_DLI, l_end_mass, u_end_mass, is_rate_degradation, c_degr