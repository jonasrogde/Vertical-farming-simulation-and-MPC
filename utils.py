import json
import numpy as np
from typing import List
    
def load_config(folder_path) -> dict:
    """Load multiple configuration files and combine them into one dictionary."""
    combined_config = {}

    file_paths = ['climate_config.json', 'crop_config.json', 'simulation_config.json']

    for file_path in file_paths:
        with open(folder_path + file_path, 'r') as file:
            config = json.load(file)
            combined_config.update(config)  # This overwrites any existing keys

    return combined_config
    
def get_attribute(data, key):
    """Get attribute from dictionary. Raise KeyError if not found."""
    if key in data:
        return data[key]
    raise KeyError(f"Key '{key}' not found in config file.")

def biomass_to_LAI(X_s, c_lar, c_tau):
    # LAI estimated in Van Henten
    return (1-c_tau)*c_lar*X_s

def LAI_to_CAC(LAI, k=0.5):
    """
    Converts Leaf Area Index (LAI) to Canopy Absorption Coefficient (CAC).

    Parameters:
    - LAI (m²/m²): Leaf Area Index, the area of leaves per unit ground area.
    - k (unitless): Extinction coefficient for the canopy (default is 0.5).

    Returns:
    - CAC (unitless): Cultivation Area Coefficient, the ratio of projected leaf area to cultivation area, indicating the proportion of the cultivation area covered by leaves.
    """
    return 1 - np.exp(-k * LAI)

def stomatal_resistance_eq(PPFD):
    """
    Calculates stomatal resistance based on Photosynthetic Photon Flux Density (PPFD).

    Parameters:
    - PPFD (μmol/m²/s): Photosynthetic Photon Flux Density, the amount of photosynthetically active photons.

    Returns:
    - Stomatal resistance (s/m): The resistance to CO2 and water vapor flux through the stomata.
    """
    return 60 * (1500 + PPFD) / (200 + PPFD)

def aerodynamical_resistance_eq(uninh_air_vel, LAI, leaf_diameter):
    """
    Calculates aerodynamic resistance based on air velocity, mean leaf diameter and Leaf Area Index (LAI).

    Parameters:
    - uninh_air_vel (m/s): Uninhibited air velocity.
    - LAI (m²/m²): Leaf Area Index, the area of leaves per unit ground area.
    -leaf_diameter (m): Mean leaf diameter.
    Returns:
    - Aerodynamic resistance (s/m): The resistance to heat and vapor flux from the canopy to the atmosphere.
    """
    return 350 * np.sqrt((leaf_diameter / uninh_air_vel)) * (1 / LAI)