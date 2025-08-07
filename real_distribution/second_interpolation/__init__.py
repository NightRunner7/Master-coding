"""
Axion Production Parameter Management
--------------------------------------

Central management for different axion production mechanisms.

Author: Krzysztof Szafrański
Date: 2025-02-12
"""

from importlib import import_module

# --- Available processes and their module paths ---
AVAILABLE_PROCESSES = {
    "electron_scattering": "fitting_function.electron_scattering",
    "muon_decay": "fitting_function.muon_decay",
    "muon_scattering": "fitting_function.muon_scattering",
    "tau_decay": "fitting_function.tau_decay",
    "tau_scattering": "fitting_function.tau_scattering",
    # "pion_decay": "fitting_function.pion_decay",   # example
}

# --- Default process ---
DEFAULT_PROCESS = "muon_scattering"


# --- Helper to load a process module ---
def get_process(process_name: str = DEFAULT_PROCESS):
    """
    Retrieve the selected axion production process module.

    Parameters
    ----------
    process_name : str
        The name of the production process.

    Returns
    -------
    module
        The module corresponding to the selected process.

    Example
    -------
    >>> process = get_process("muon_scattering")
    >>> funcs, coeffs = process.function_candidates, process.best_coeffs
    >>> A_val = funcs["A"]["polynomial"](1.0, *coeffs["A"]["polynomial"])
    """
    if process_name not in AVAILABLE_PROCESSES:
        raise ValueError(f"Invalid process '{process_name}'. "
                         f"Available: {list(AVAILABLE_PROCESSES.keys())}")

    return import_module(AVAILABLE_PROCESSES[process_name])


# --- Convenience: list all processes ---
def list_available_processes():
    """Return a list of available axion production processes."""
    return list(AVAILABLE_PROCESSES.keys())
