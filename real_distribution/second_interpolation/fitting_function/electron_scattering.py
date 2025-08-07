"""
Best-Fit Parameters and Functions for Electron Scattering
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
        4.64825837e-05,  2.67727520e-04, -6.20228993e-06, -2.30679063e-03,
        -2.07762855e-03,  7.52840699e-03,  8.53338690e-03, -1.16561248e-02,
        -1.33821356e-02,  6.42630101e-03,  3.95346494e-03, -6.35326766e-03,
        -6.19536448e-03,  1.47110879e+00
    ],
    "b": [
        2.34241002e-05,  8.99165220e-05, -1.73405139e-04, -7.87886172e-04,
        6.37398203e-04,  2.61658578e-03, -1.86221722e-03, -5.14393324e-03,
        -2.18572469e-03, -1.13846927e-02, -2.85792704e-02, -3.84039786e-02,
        1.96047776e+00, -3.12399008e+00
    ],
    "mu": [
        9.05527356e-02, -4.81601597e-02, -2.47581175e+00, -1.39402807e+00,
        1.79583690e+01,  1.08459090e+01, -5.31339044e+01, -3.49952362e+01,
        9.98533383e+01, -2.66243186e+01,  5.69794088e+01, -1.63605315e+02,
        1.64669990e+02, -7.97218732e+01
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
