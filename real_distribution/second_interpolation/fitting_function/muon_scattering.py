"""
Best-Fit Parameters and Functions for Muon Scattering
------------------------------------------------------

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

# ################################# FUNCTION WHICH WORKS ############################################################# #
# --- Best-fit coefficients for functional models ---
best_function_coeffs = {
    "piecewise": [
        1.25540634, 25.23140449, -213.21252559, -1.50216418, -176.84370719,
        -40.06637671, 245.14623916, 21.13818352, 16.5282045, -186.7395364,
        -33.52252315, -161.47819936, -8.70319333, 219.62748638, 18.73939703, -2.0
    ],
    "b_function": [
        -2.77576175, 1.82940840, 2.22230261, 1.63986335, -6.42279540e-02,
        -2.37363679e-04, -1.46742969e-03, -1.62412865e-05, 1.48303556e-02,
        1.50951749e-02, -6.37351025e-02, -8.46885428e-02, 1.77349302e-01,
        2.40159283e-01, -6.27056763e-01, 3.90298915e-01, -1.09621190e-01
    ],
    "exp_decay": [-0.89232017, 0.49383594, -2.12668891]
}

# --- Best polynomial coefficients ---
best_polynomial_coeffs = {
    "A": [
        1.56559417e-05,  1.44665366e-04,  3.18263373e-04, - 8.18634807e-04,
        -3.82284830e-03, -1.85593688e-04,  1.52036557e-02,  1.00049709e-02,
        -3.14890853e-02, -2.52676918e-02,  4.90630469e-02,  2.36040527e-02,
        -1.19228329e-01,  1.11060238e+00
    ],
    "b": [
        4.70494646e-05,  3.70545103e-04,  4.38570979e-04, -3.17235523e-03,
        -7.20565091e-03,  1.15323140e-02,  3.47399996e-02, -3.25267254e-02,
        -9.78207471e-02,  1.12940592e-01,  2.30144764e-01, -5.63877245e-01,
        3.78121904e-01, - 1.31034922e-01
    ],
    "mu": [
        1.80703572e-03,  1.15221881e-02,  4.68544104e-03, -9.57267829e-02,
        -1.21018225e-01,  2.92556553e-01,  4.87289678e-01, -5.42438144e-01,
        -4.47287960e-01, -2.07847459e-01,  1.76562466e+00, -2.14455439e+00,
        1.26903374e+00, -2.87998071e+00
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
        "piecewise": best_function_coeffs["piecewise"],
        "polynomial": best_polynomial_coeffs["A"]
    },
    "b": {
        "b_function": best_function_coeffs["b_function"],
        "polynomial": best_polynomial_coeffs["b"]
    },
    "mu": {
        "exp_decay": best_function_coeffs["exp_decay"],
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
