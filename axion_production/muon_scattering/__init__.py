"""
Muon Scattering Module for Axion Production
--------------------------------------------

This module defines the parameter dependencies for axion production via muon scattering.
It provides fitting functions and best-fit parameters for A(m_a), b(m_a), and μ(m_a).

Loaded Components:
- `fitting_functions.py`: Defines different fitting functions for each parameter
and Stores best-fit parameters for selected functions.

Functionality:
- `DEFAULT_FITTING_FUNCTIONS`: The standard default function choices.
- `set_fitting_function(param_type, function_name)`: Dynamically change the function fit.
- `evaluate_parameter(param_type, x)`: Evaluate A, b, or μ at a given x.

Usage:
- Import this module to access axion production fitting functions.

Author: Krzysztof Szafrański
Date: 2025-02-16
"""

# Import necessary components
from .fitting_functions import function_candidates, best_coeffs

# --- Keep default functions immutable ---
DEFAULT_FITTING_FUNCTIONS = {
    "A": function_candidates["A"]["polynomial"],
    "b": function_candidates["b"]["polynomial"],
    "mu": function_candidates["mu"]["polynomial"]
}

DEFAULT_COEFFS = {
    "A": best_coeffs["A"]["polynomial"],
    "b": best_coeffs["b"]["polynomial"],
    "mu": best_coeffs["mu"]["polynomial"]
}

# --- Mutable dictionary to allow dynamic selection ---
selected_fitting_functions = DEFAULT_FITTING_FUNCTIONS.copy()

# --- Map best-fit parameters from best_coeffs ---
selected_coeffs = DEFAULT_COEFFS.copy()

# --- Function to modify function selection dynamically ---
def set_fitting_function(param_type, function_name):
    """
    Change the fitting function for a parameter.

    Parameters:
        param_type (str): "A", "b", or "mu".
        function_name (str): Name from `function_candidates`, or "default" to reset.

    Example:
    set_fitting_function("A", "rational_6th")
    set_fitting_function("b", "default")
    """
    if param_type not in selected_fitting_functions:
        raise ValueError(f"Invalid parameter type: {param_type}. Choose from {list(selected_fitting_functions.keys())}.")

    if function_name == "default":
        selected_fitting_functions[param_type] = DEFAULT_FITTING_FUNCTIONS[param_type]
        selected_coeffs[param_type] = DEFAULT_COEFFS[param_type]
        print(f"Reset {param_type} fitting function to default.")
        return

    if function_name not in function_candidates[param_type]:
        print(f"Invalid function name '{function_name}' for {param_type}. Available options:")
        for fn_name in function_candidates[param_type].keys():
            print(f"  - {fn_name}")
        return

    selected_fitting_functions[param_type] = function_candidates[param_type][function_name]
    selected_coeffs[param_type] = best_coeffs[param_type][function_name]
    print(f"Set {param_type} fitting function to: {function_name}")

# --- Function to evaluate parameters ---
def evaluate_parameter(param_type, x):
    """Evaluate the chosen fitting function for a parameter at x."""
    if param_type not in selected_fitting_functions:
        raise ValueError(f"Invalid parameter type: {param_type}.")

    func = selected_fitting_functions[param_type]
    if func is None:
        raise ValueError(f"No function set for {param_type}. Currently selected: {selected_fitting_functions[param_type]}."
                         f" Reset with `set_fitting_function('{param_type}', 'default')`.")

    params = selected_coeffs[param_type]
    return func(x, *params)
