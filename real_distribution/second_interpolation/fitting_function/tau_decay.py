"""
Best-Fit Parameters and Functions for Tau Scattering
---------------------------------------------------------

This module provides a structured approach to fitting axion production parameters
(A, b, μ) as functions of axion mass (m_a). It includes:

- **Functional Approximations:** Rational, logistic, exponential decay, and piecewise functions.
- **Polynomial Fits:** Best polynomial fits determined from numerical analysis.
- **Best-Fit Coefficients:** Stored separately for functional and polynomial models.
- **Dynamic Function Selection:** A dictionary `function_candidates` maps function names to implementations.

Usage:
- Import this module in `muon_scattering/__init__.py` to access fitting functions.
- The `best_coeffs` dictionary provides the best available coefficients.

Author: Krzysztof Szafrański
Date: 2025-02-16
"""

import numpy as np

# ################################# FITTING FUNCTIONS ################################################################ #
# --- A PARAMETER FUNCTIONS --------------------------------------------------------------------------------------------
def rational_function_6th_order(x, a6, a5, a4, a3, a2, a1, a0, b6, b5, b4, b3, b2, b1, b0):
    """6th-order rational function for fitting A(m_a)."""
    numerator = a6 * x**6 + a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0
    denominator = b6 * x**6 + b5 * x**5 + b4 * x**4 + b3 * x**3 + b2 * x**2 + b1 * x + b0
    return numerator / denominator

def modified_rational_function(x, a6, a5, a4, a3, a2, a1, a0, b6, b5, b4, b3, b2, b1, b0, c):
    """Modified 6th-order rational function with a plateau constant term."""
    return rational_function_6th_order(x, a6, a5, a4, a3, a2, a1, a0, b6, b5, b4, b3, b2, b1, b0) + c

def piecewise_function(x, c, a6, a5, a4, a3, a2, a1, a0, b6, b5, b4, b3, b2, b1, b0, x_switch):
    """Piecewise function combining a plateau and a rational function."""
    rational_part = rational_function_6th_order(x, a6, a5, a4, a3, a2, a1, a0, b6, b5, b4, b3, b2, b1, b0)
    return np.where(x < x_switch, c, rational_part)

# --- B PARAMETER FUNCTIONS --------------------------------------------------------------------------------------------
def b_function(x, x_min, x_max, a, b_intercept, b_max, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
    """
    Computes b(x) with:
    - A linear function for x < x_min.
    - A constant value b_max for x > x_max.
    - An 11th-degree polynomial for intermediate values.
    """
    x = np.asarray(x)
    polynomial = np.poly1d([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11])
    return np.where(
        x > x_max, b_max,
        np.where(x < x_min, a * x + b_intercept, polynomial(x))
    )

def refined_logistic_function(x, L, k, x0, a1, a2, plateau):
    """Refined logistic function with additional transition control terms."""
    logistic = L / (1 + np.exp(-(k * (x - x0) + a1 * x + a2 * x**2)))
    return logistic + plateau

# --- MU PARAMETER FUNCTIONS -------------------------------------------------------------------------------------------
def exponential_decay(x, a, b, c):
    """Exponential decay function: a * exp(-x/b) + c"""
    return a * np.exp(-x / b) + c

# ################################# POLYNOMIAL FIT FUNCTIONS ######################################################### #
def polynomial_fit_A(x, *coeffs):
    """Polynomial fit function for A(m_a)."""
    return np.polyval(coeffs, x)

def polynomial_fit_b(x, *coeffs):
    """Polynomial fit function for b(m_a)."""
    return np.polyval(coeffs, x)

def polynomial_fit_mu(x, *coeffs):
    """Polynomial fit function for μ(m_a)."""
    return np.polyval(coeffs, x)

# ################################# STORED BEST-FIT COEFFICIENTS ##################################################### #
# --- Best-fit coefficients for functional models ---
# NONE FUNCTION HAVE BEEN TESTED YET
best_function_coeffs = {}

# --- Best polynomial coefficients ---
best_polynomial_coeffs = {
    "A": [
        4.19502265e-09, -1.89480204e-08, -3.94336853e-07,  1.21268457e-06,
        1.27618911e-05, -2.83137942e-05, -1.78224382e-04,  2.68657363e-04,
        1.23450571e-03, -1.15723292e-03, -7.73928815e-03,  2.56853685e-02,
        -2.30616454e-02,  8.85915881e-01
    ],
    "b": [
        -1.34397916e-08, -2.16847812e-07,  7.16195445e-07,  1.36910615e-05,
        -1.87108667e-05, -3.05466220e-04,  3.29844632e-04,  2.70847467e-03,
        -2.13416371e-03, -1.08181472e-02, -6.28518910e-03,  7.78043190e-02,
        5.23675673e-02, -8.78740749e-01
    ],
    "mu": [
        -2.80545901e-08, -8.47050035e-07,  2.95459337e-07,  4.66088822e-05,
        4.91808479e-05, -1.00277288e-03, -1.03607150e-03,  8.83880246e-03,
        1.09261336e-02, -4.05068554e-02, -9.42081656e-02,  2.91028247e-01,
        5.87244208e-01, -5.39458198e+00
    ]
}

# ################################# FUNCTION WHICH WORKS ############################################################# #
# --- Dictionary mapping function names to their respective implementations ---
# This dictionary stores all available fitting functions for each parameter (A, b, μ).
# Each parameter has multiple fitting options, including polynomial, rational, or exponential functions.
function_candidates = {
    "A": {
        "rational_6th": rational_function_6th_order,
        "modified_rational": modified_rational_function,
        "piecewise": piecewise_function,
        "polynomial": polynomial_fit_A
    },
    "b": {
        "b_function": b_function,
        "logistic": refined_logistic_function,
        "polynomial": polynomial_fit_b
    },
    "mu": {
        "exp_decay": exponential_decay,
        "polynomial": polynomial_fit_mu
    }
}

# --- Dictionary mapping the best-fit coefficients for each function ---
# This dictionary stores the **precomputed** best-fit parameter values for each function.
# These values were obtained from numerical fitting and should be used for evaluation.
best_coeffs = {
    "A": {
        "polynomial": best_polynomial_coeffs["A"]
    },
    "b": {
        "polynomial": best_polynomial_coeffs["b"]
    },
    "mu": {
        "polynomial": best_polynomial_coeffs["mu"]
    }
}

# ################################# FUNCTION SELECTION ############################################################### #
# --- Defaults (immutable) ---
DEFAULT_FITTING_FUNCTIONS = {
    "A": function_candidates["A"]["polynomial"],
    "b": function_candidates["b"]["polynomial"],
    "mu": function_candidates["mu"]["polynomial"],
}

DEFAULT_COEFFS = {
    "A": best_coeffs["A"]["polynomial"],
    "b": best_coeffs["b"]["polynomial"],
    "mu": best_coeffs["mu"]["polynomial"],
}

# --- Mutable copies ---
selected_fitting_functions = DEFAULT_FITTING_FUNCTIONS.copy()
selected_coeffs = DEFAULT_COEFFS.copy()

# --- Per-process API ---
def set_fitting_function(param_type: str, function_name: str):
    """
    Set or reset the fitting function for a given parameter (A, b, mu).
    """
    if param_type not in selected_fitting_functions:
        raise ValueError(f"Invalid parameter type: {param_type}")

    if function_name == "default":
        selected_fitting_functions[param_type] = DEFAULT_FITTING_FUNCTIONS[param_type]
        selected_coeffs[param_type] = DEFAULT_COEFFS[param_type]
        return

    if function_name not in function_candidates[param_type]:
        raise ValueError(f"Invalid function '{function_name}' for {param_type}. "
                         f"Available: {list(function_candidates[param_type].keys())}")

    selected_fitting_functions[param_type] = function_candidates[param_type][function_name]
    selected_coeffs[param_type] = best_coeffs[param_type][function_name]


def evaluate_parameter(param_type: str, x):
    """
    Evaluate the currently selected function for a parameter at value x.
    """
    if param_type not in selected_fitting_functions:
        raise ValueError(f"Invalid parameter type: {param_type}")

    func = selected_fitting_functions[param_type]
    coeffs = selected_coeffs[param_type]
    return func(x, *coeffs)
