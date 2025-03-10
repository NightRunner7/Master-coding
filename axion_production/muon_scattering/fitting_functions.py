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

# ################################# STORED BEST-FIT COEFFICIENTS ##################################################### #
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
        1.56135666e-05,  1.44420277e-04,  3.18298827e-04, -8.16379403e-04,
        -3.82106382e-03, -1.93658245e-04,  1.51956535e-02,  1.00191846e-02,
        -3.14759855e-02, -2.52802868e-02,  4.90542769e-02,  2.36086577e-02,
        -1.19226303e-01,  1.11060218e+00
    ],
    "b": [
        4.69102737e-05,  3.69749467e-04,  4.38747928e-04, -3.16490973e-03,
        -7.20023348e-03,  1.15051454e-02,  3.47148619e-02, -3.24776595e-02,
        -9.77788267e-02,  1.12895896e-01,  2.30115920e-01, -5.63860461e-01,
        3.78128998e-01, -1.31035627e-01
    ],
    "mu": [
        1.75788438e-03,  1.12241762e-02,  4.63418701e-03, -9.31838169e-02,
        -1.18283009e-01,  2.84356283e-01,  4.76374332e-01, -5.29909079e-01,
        -4.30764344e-01, -2.17040310e-01,  1.75594817e+00, -2.14183745e+00,
        1.27055755e+00, -2.88012442e+00
    ]


    # "A": [
    #     -1.10941982e-05, -1.69369835e-04, -6.47086049e-04,  3.56957005e-04,
    #     5.69394653e-03,  3.73533619e-03, -1.94066519e-02, -1.58532270e-02,
    #     4.27507213e-02,  1.94556353e-02, -1.18259546e-01,  1.11087253e+00
    # ],
    # "b": [
    #     -1.16930210e-04, -8.60348558e-04, -6.87383009e-04,  7.92833223e-03,
    #     1.33447050e-02, -3.49498377e-02, -6.82680572e-02,  1.21610557e-01,
    #     2.13437448e-01, -5.68293111e-01,  3.80826282e-01, -1.30743745e-01
    # ],
    # "mu": [
    #     1.61203369e-03,  7.96925026e-03,  1.76552702e-03, -6.42538011e-02,
    #     -3.00309439e-02,  5.04638606e-02,  3.98182874e-01, -6.94794859e-01,
    #     1.22366480e+00, -1.97794898e+00,  1.36313079e+00 -2.89053176e+00
    # ]

    # "A": [
    #     -2.86063803e-05, -2.69799110e-04, -5.17166975e-04, 1.91388372e-03,
    #     6.64291681e-03, -3.58895695e-03, -2.76759519e-02, -1.82558394e-03,
    #     6.22438714e-02, 8.02587869e-03, -1.38904014e-01, 1.12831659e+00
    # ],
    # "b": [
    #     -2.18517724e-04, -1.36453392e-03, -3.36731798e-08, 1.41375347e-02,
    #     1.44209933e-02, -6.21440130e-02, -8.28085221e-02, 1.75891212e-01,
    #     2.38510373e-01, -6.26592196e-01, 3.90659728e-01, -1.09645635e-01
    # ],
    # "mu": [
    #     7.85008268e-04, 1.14985041e-03, -5.16498990e-03, -1.53589708e-02,
    #     5.02540035e-02, -7.84486470e-02, 1.78973632e-01, -5.66151799e-01,
    #     1.48061388e+00, -2.08054671e+00, 1.29694376e+00, -2.81321784e+00
    # ]
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
