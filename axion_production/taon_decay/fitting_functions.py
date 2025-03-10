"""
Best-Fit Parameters and Functions for Taon Decays
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
    # "A": [
    # 1.00192879e-07,  2.61125459e-06,  2.68746380e-05,  1.29727058e-04,
    # 2.11842419e-04, -5.78654254e-04, -2.63074163e-03, -1.04905399e-03,
    # 8.03250941e-03,  7.96087932e-03, -9.70972238e-03, -1.10866798e-02,
    # -6.15971229e-04,  2.76580101e-02, -2.48689826e-02,  8.86356626e-01
    # ],
    # "b": [
    # 4.68783009e-07,  1.14587321e-05,  1.08545806e-04,  4.59648050e-04,
    # 4.63125969e-04, -2.86736764e-03, -8.39847796e-03,  2.21540332e-03,
    # 2.95653984e-02,  1.20595062e-02, -4.15695486e-02, -2.09246417e-02,
    # 8.38909788e-03,  7.03162645e-02,  5.73686307e-02, -8.75842945e-01
    # ],
    # "mu": [
    # 8.39786068e-07,  2.02929291e-05,  1.88296861e-04,  7.59149903e-04,
    # 5.20851511e-04, -5.57356090e-03, -1.32476708e-02,  1.02268742e-02,
    # 5.07065923e-02, -2.22078403e-03, -7.06699877e-02,  1.42060676e-03,
    # -4.40277044e-02,  2.18339649e-01,  5.82467149e-01, -5.37850726e+00
    # ]

    "A": [
        -5.26011517e-07, -1.08512129e-05, -8.19593243e-05, -2.34857772e-04,
        1.35636050e-04,  1.96047177e-03,  1.85303673e-03, -5.22561551e-03,
        -6.48428819e-03,  6.66247147e-03,  1.33019291e-03,  1.96029611e-02,
        -2.58413487e-02,  8.86955338e-01
    ],
    "b": [
        -1.43731365e-06, -2.56856619e-05, -1.49927812e-04, -1.59807821e-04,
        1.36117512e-03,  3.39266408e-03, -4.25678086e-03, -1.55813760e-02,
        7.26581623e-03,  2.86365779e-02, -2.26668323e-02,  4.43921246e-02,
        6.40904636e-02, -8.73872330e-01
    ],
    "mu": [
        -3.09985162e-06, -5.70195516e-05, -3.42085093e-04, -3.83645090e-04,
        3.25231513e-03,  8.34912879e-03, -1.13999197e-02, -4.12152141e-02,
        2.72400780e-02,  7.98604075e-02, -1.12070110e-01,  1.75533235e-01,
        5.97991132e-01, -5.37523370e+00
    ]
}

# ################################# FUNCTION SELECTION ############################################################### #
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
