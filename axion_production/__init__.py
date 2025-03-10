"""
Axion Production Parameter Management
--------------------------------------

This module provides a structured way to manage different axion production mechanisms
by dynamically loading the necessary fitting functions and precomputed parameters.

Loaded Components:
- `muon_scattering`: Module handling axion production via muon scattering.
- (Future modules like `pion_decay` can be added similarly.)

Functionality:
- `AVAILABLE_PROCESSES`: A dictionary that maps process names to their respective modules.
- `get_process(process_name)`: Allows dynamic selection of a production process.
- `list_available_processes()`: Lists all available production processes.

Usage:
- Import this module to access different axion production scenarios dynamically.

Author: Krzysztof Szafrański
Date: [2025-02-12]
"""

# Import available production processes
from . import muon_scattering  # Import the muon scattering module
from . import electron_scattering  # Import the electron scattering module
from . import taon_decay  # Import the taon decay module
# from . import pion_decay  # Future process (example)

# --- Map process names to their respective modules ---
AVAILABLE_PROCESSES = {
    "muon_scattering": muon_scattering,
    "electron_scattering": electron_scattering,
    "taon_decay": taon_decay
    # "pion_decay": pion_decay,  # Future process
}


# --- Function to dynamically retrieve a process ---
def get_process(process_name):
    """
    Retrieve the selected axion production process module.

    Parameters:
    - process_name (str): The name of the production process.

    Returns:
    - The module corresponding to the selected process.

    Example:
    ```
    process = get_process("muon_scattering")
    A_value = process.evaluate_parameter("A", -3.0)
    ```
    """
    if process_name not in AVAILABLE_PROCESSES:
        raise ValueError(f"Invalid process name: {process_name}. Use list_available_processes() to see options.")

    return AVAILABLE_PROCESSES[process_name]


# --- Function to list available production mechanisms ---
def list_available_processes():
    """
    List all available axion production processes.

    Returns:
    - List of available process names.

    Example:
    ```
    print(list_available_processes())
    ```
    """
    return list(AVAILABLE_PROCESSES.keys())