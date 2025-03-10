"""
Taon Decay Axion Distribution and ΔN_eff Computation
----------------------------------------------------------

This module provides tools for analyzing the axion distribution function
resulting from **taon decay production**. It builds upon the interpolated
distribution data and utilizes fitted parameter dependencies (A, b, μ) to
reconstruct the axion distribution. Additionally, it computes the axion
contribution to the extra relativistic degrees of freedom (ΔN_eff).

### Key Functionalities:
1. **Interpolation Handling**
   - Inherits from `FirstInterpolation` to process initial distribution data.
   - Implements `generate_second_interpolation()` to reconstruct the best-fit
     axion distribution for any axion mass.

2. **Fitting Function Integration**
   - Uses `evaluate_parameter()` from `taon_decay` to obtain best-fit
     parameters for A(m_a), b(m_a), and μ(m_a).
   - Supports switching between different fitting methods (polynomials, rational
     functions, etc.).

3. **ΔN_eff Computation**
   - Implements `calculate_delta_n_eff()` to determine the axion contribution
     to relativistic degrees of freedom.
   - Includes entropy ratio scaling to account for axion decoupling conditions.

### Methods:
- **generate_second_interpolation(axion_mass, q_arr, dist)**
  Generates the reconstructed axion distribution using best-fit parameters.

- **calculate_delta_n_eff(m_a, parameters_arr=None)**
  Computes the axion contribution to ΔN_eff via numerical integration.

### Usage Example:
```python
# Load and initialize the axion distribution class
taon_decay = TaonDecayModel("data/axion_distributions.dat")

# Generate interpolated distribution
q_values = np.linspace(0.01, 20, 100)
axion_mass = 1e-3  # eV
distribution = taon_decay.generate_second_interpolation(axion_mass, q_values)

# Compute ΔN_eff for a given axion mass
delta_N_eff = taon_decay.calculate_delta_n_eff(axion_mass)
"""

import numpy as np
from scipy.integrate import quad
# --- FROM EXTERNAL FILES ---
from axion_production.taon_decay import evaluate_parameter
from relativistic_degrees_of_freedom import RelativisticDegreesOfFreedom
from axion_production.distribution_first_interpolation import FirstInterpolation

class TaonDecayModel(FirstInterpolation):
    """
    Class for handling axion distribution functions from taon decay production.

    This class:
    - Loads axion distribution data from an input file.
    - Performs interpolation of the distributions (via inherited `FirstInterpolation`).
    - Computes second interpolation for reconstructed distributions.
    - Provides a function to compute the extra relativistic degrees of freedom (ΔN_eff).

    Attributes:
    -----------
    con : dict
        Stores physical constants and degree-of-freedom ratios.

    Methods:
    --------
    generate_second_interpolation(axion_mass, q_arr, dist):
        Generates a reconstructed distribution using stored best-fit parameters.

    calculate_delta_n_eff(m_a, parameters_arr=None, output_params=False):
        Computes ΔN_eff from the axion distribution.
    """

    # ----------------------------------------- INITIALIZATION ------------------------------------------------------ #
    def __init__(self, file_path):
        """
        Initialize axion model class for taon decay.

        Parameters:
        -----------
        file_path : str
            Path to the data file containing axion distributions.
        """
        # --- Inherit from FirstInterpolation (load data and perform first interpolate of distributions) ---
        super().__init__(file_path)

        # --- Relativistic Degrees of Freedom Handler ---
        RelativisticDOF = RelativisticDegreesOfFreedom()

        # --- Physical Constants ---
        self.con = dict()  # dictionary with constants
        self.con["taon_mass"] = 1777*10**6  # [eV]
        self.con["kB_T_today"] = 8.617 * 10**(-5) * 2.725  # kB * T, [kB*T] = [kB] * [T] = [eV/K] * [K] = [eV]
        self.con["x_dec"] = 30  # Axion decoupling scale
        self.con["g_dof_axion"] = 1  # axion degrees of freedom
        self.con["g_dof_photon"] = 2  # photon degrees of freedom

        # --- Compute Relativistic Degrees of Freedom Ratios ---
        self.con["g_star_s_today"] = RelativisticDOF.get_degrees_of_freedom(self.con["kB_T_today"], dof_type="g_eff_s")
        self.con["g_star_s_axion_decoupling"] = RelativisticDOF.compute_decoupling_dof(self.con["taon_mass"], self.con["x_dec"])

        # --- Limits ---
        self.con["q_max"] = 19.99

    # ----------------------------------------- APPROXIMATE DISTRIBUTIONS -------------------------------------------- #
    @staticmethod
    def f_approx_q2(q, A, b, mu):
        """
        Approximate function for fitting f(q) * q^2.

        Returns:
            Evaluated function values at q.
        """
        # return np.sqrt(q ** 2 + 1) * q ** 1 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1
        # return q ** 1 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1
        return q ** 2 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1

    @staticmethod
    def f_approx_q2_adam(q, A, b, mu):
        """
        Approximate function for fitting f(q) * q^2. Almost Adam's version.

        Returns:
            Evaluated function values at q.
        """
        return np.sqrt(q ** 2 + 1) * q ** 1 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1

    # ----------------------------------------- SECOND INTERPOLATION DISTRIBUTION ------------------------------------ #
    def generate_second_interpolation(self, axion_mass, q_arr, dist="f(q)_q2"):
        """
        Generate the second interpolation of the axion distribution.

        This function computes the interpolated distribution using the best-fit parameters
        for a given axion mass. It retrieves the fitted A(m_a), b(m_a), and μ(m_a) values
        and reconstructs the approximate distribution function.

        Parameters:
        - axion_mass (float): The axion mass in eV.
        - q_arr (np.ndarray): Array of co-moving momenta q values.
        - dist (str): The type of distribution to reconstruct.
            Options: "f(q)", "f(q)_q1", "f(q)_q2", "f(q)_q3".

        Returns:
        - np.ndarray: The reconstructed second interpolation of the selected distribution.
        """

        # Compute log(m_a) for function evaluation
        log_axion_mass = np.log(axion_mass)

        # Retrieve interpolated fit parameters for given mass
        A = evaluate_parameter("A", log_axion_mass)
        b = evaluate_parameter("b", log_axion_mass)
        mu = evaluate_parameter("mu", log_axion_mass)

        # Compute the corresponding reconstructed distribution
        reconstructed_distribution = self.f_approx_q2(q_arr, A, b, mu)

        # Adjust based on selected distribution type
        distribution_scaling = {
            "f(q)_q3": q_arr,
            "f(q)_q2": 1,
            "f(q)_q1": q_arr ** (-1),
            "f(q)": q_arr ** (-2),
        }

        if dist in distribution_scaling:
            return reconstructed_distribution * distribution_scaling[dist]

        raise ValueError(f"Invalid distribution type: {dist}. Choose from {list(distribution_scaling.keys())}.")

    # ----------------------------------------- CALCULATE DELTA N_eff ------------------------------------------------ #
    def calculate_delta_n_eff(self, m_a, parameters_arr=None):
        """
        Compute the extra relativistic degrees of freedom (ΔN_eff) for axion mass m_a.

        Parameters:
        -----------
        - m_a (float): Axion mass in eV.
        - parameters_arr (dict, optional): If provided, manually supply A, b, μ instead of computing them.

        Returns:
        --------
        float: The contribution of axions to ΔN_eff.
        """
        # --- Physical calculations
        # Ratio of entropy degrees of freedom today to late axion decoupling
        g_star_s_ratio = self.con["g_star_s_axion_decoupling"] / self.con["g_star_s_today"]
        # Ratio of axion and photon degrees of freedom
        g_dof_ratio = self.con["g_dof_axion"] / self.con["g_dof_photon"]

        # Compute constant pre-factor for ΔN_eff
        const = 8 / 7 * (11 / 4) ** (4 / 3) * 15 / (np.pi ** 4) * g_dof_ratio * g_star_s_ratio ** (-4 / 3)

        # --- Deal with Distribution
        # Retrieve fitted parameters
        param_A, param_b, param_mu = self.get_fitted_parameters(m_a, parameters_arr)

        # Define the distribution function
        def distribution(q_val):
            """
            Computes the approximated axion distribution function in terms of q_tilda.
            """
            # Compute the approximate distribution value using fitted parameters
            return self.f_approx_q2(q_val, param_A, param_b, param_mu) * q_val

        # Perform numerical integration
        integral_of_dist, _ = quad(distribution, 0, self.con["q_max"])

        return const * integral_of_dist

    # ----------------------------------------- GET DATA ------------------------------------------------ #
    @staticmethod
    def get_fitted_parameters(m_a, parameters_arr=None):
        """
        Retrieve the fitted parameters A, b, and μ for a given axion mass.

        Parameters:
        -----------
        - m_a (float): Axion mass in eV.
        - parameters_arr (dict, optional): If provided, manually supply A, b, μ instead of computing them.

        Returns:
        --------
        tuple: (A, b, μ) best-fit parameters for the given axion mass.
        """
        if parameters_arr:
            return parameters_arr["A"], parameters_arr["b"], parameters_arr["mu"]

        log_m_a = np.log(m_a)
        return (
            evaluate_parameter("A", log_m_a),
            evaluate_parameter("b", log_m_a),
            evaluate_parameter("mu", log_m_a)
        )

    def get_q_tilda_upper_limit(self):
        return self.con["q_max"] * (self.con["g_star_s_axion_decoupling"] / self.con["g_star_s_today"]) ** (1 / 3)
