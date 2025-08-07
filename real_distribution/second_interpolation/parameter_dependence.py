"""
Axion Parameter Dependence Analysis
-----------------------------------

This module provides a structured approach to studying the dependence of
parameters A(m_a), b(m_a), and μ(m_a) on axion mass m_a.

Features:
- Loads best-fit parameters from a selected production process.
- Performs interpolation and functional fitting for parameter dependence.
- Supports both functional and polynomial fits.

Author: Krzysztof Szafrański
Date: 2025-02-12
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
# --- FROM EXTERNAL FILES ---
from second_interpolation import get_process
from distribution_first_interpolation import FirstInterpolation

class ParametersAxionMassDependence(FirstInterpolation):
    """
    A class for analyzing the dependence of axion distribution parameters on axion mass.

    Functionality:
    - Loads best-fit parameters from a selected production process.
    - Interpolates extracted data and fits functional dependencies.
    - Provides access to fitted functions for A(m_a), b(m_a), and μ(m_a).
    """
    def __init__(self, process_name, file_path):
        """
        Initialize the class with a specific axion production process.

        Parameters:
        - process_name: The selected axion production process module (e.g., muon_scattering).
        - file_path (str): Path to the input data file.
        """
        super().__init__(file_path)

        # --- Load production process dynamically (module with stateful API) ---
        self.process = get_process(process_name)

        # --- Store available fitting functions ---
        self.fitting_functions = {
            "best": self._wrap_process_defaults(),
            "default": self._wrap_process_defaults(),
            "test_polynomial": None,
            "test_fit": None
        }

        # Initialize storage for fitted parameters (unchanged)
        self.param["fit"] = {
            dist: {"A": None, "b": None, "mu": None}
            for dist in ["f(q)", "f(q)_q1", "f(q)_q2", "f(q)_q3"]
        }

        # Fit range settings or
        self.param["fit"]["q_min"] = 0.01
        self.param["fit"]["q_max"] = 19.99
        self.param["fit"]["q_step"] = 0.01

    def _wrap_process_defaults(self):
        """
        Return a dictionary of callables bound to the process module's defaults.
        These callables automatically use the process-selected coefficients.
        """
        return {
            param: lambda x, param=param: self.process.evaluate_parameter(param, x)
            for param in ["A", "b", "mu"]
        }

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
        # return np.sqrt(q ** 2 + 1) * q ** 2 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1

    @staticmethod
    def f_approx_q2_adam(q, A, b, mu):
        """
        Approximate function for fitting f(q) * q^2. Almost Adam's version.

        Returns:
            Evaluated function values at q.
        """
        return np.sqrt(q ** 2 + 1) * q ** 1 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1

    ####################################################################################################################
    ########################################### COMPARE TO DISTRIBUTION ################################################
    ####################################################################################################################
    # ----------------------------------------- FITTING DISTRIBUTION DATA -------------------------------------------- #
    def fit_single_distribution(self, index, dist="f(q)_q2"):
        """
        Fit an approximate function to an interpolated distribution.

        --------------------------------------------------------
        Info:

        result: tuple = curve_fit(model_functions[dist], q_values, data_to_fit, method='lm', maxfev = 5000)

        Is the best at now. But it does not work properly with tau scatterings due to change the normalisation
        factor between two .data files (I believe this is issue)
        --------------------------------------------------------

        Parameters:
            index (int): Index of the distribution.
            dist (str): Distribution type.

        Returns:
            dict: Best-fit parameters {A, b, mu}.
        """
        N = self.distributions["input"]["N"]
        if index >= N:
            raise IndexError(f"Invalid index {index}. Available range: 0 ≤ index < {N}.")

        # Generate q values
        q_values = np.arange(self.param["fit"]["q_min"], self.param["fit"]["q_max"], self.param["fit"]["q_step"])

        # Evaluate interpolated distribution
        interp_func = self.distributions["interp1"][dist][index]
        data_to_fit = interp_func(q_values)

        # Select the correct approximate function
        model_functions = {
            "f(q)_q3": lambda q, A, b, mu: self.f_approx_q2(q, A, b, mu) * q,
            "f(q)_q2": lambda q, A, b, mu: self.f_approx_q2(q, A, b, mu),
            "f(q)_q1": lambda q, A, b, mu: self.f_approx_q2(q, A, b, mu) * q ** (-1),
            "f(q)": lambda q, A, b, mu: self.f_approx_q2(q, A, b, mu) * q ** (-2)
        }

        if dist not in model_functions:
            raise ValueError(f"Invalid distribution type: {dist}")


        if index > 0 and self.param["fit"][dist]["A"] is not None:
            # Warm start from previous result
            prev = self.param["fit"][dist]
            initial_guess = [
                prev["A"][index - 1],
                prev["b"][index - 1],
                prev["mu"][index - 1]
            ]
        else:
            # Data-informed guess
            A_guess = max(0.5, np.mean(data_to_fit) / (np.max(q_values) + 1))  # scale with distribution
            b_guess = np.log(np.max(data_to_fit) + 1)  # scale with height
            mu_guess = max(0.01, np.min(data_to_fit))  # avoid zero or negative
            initial_guess = [A_guess, b_guess, mu_guess]

        # --- Select weights
        # weights = 1.0 * (1.0 + q_values ** 2)  # higher weight at high q
        # weights = np.log(1 + q_values)
        weights = (1.0 + q_values ** 2) / (1.0 + q_values)

        # --- Perform the fit
        # result: tuple = curve_fit(model_functions[dist], q_values, data_to_fit, method='trf')
        result: tuple = curve_fit(model_functions[dist], q_values, data_to_fit, method='lm')
        # result: tuple = curve_fit(model_functions[dist], q_values, data_to_fit,
        #                        p0=initial_guess, method="lm", sigma=weights)
        popt, pcov = result
        return {"A": popt[0], "b": popt[1], "mu": popt[2]}

    def fit_all_distributions(self, dist="f(q)_q2"):
        """Fit all distributions in the dataset and store results progressively."""
        N = self.distributions["input"]["N"]

        # Prepare empty lists to accumulate results
        A_vals, b_vals, mu_vals = [], [], []

        for i in range(N):
            result = self.fit_single_distribution(i, dist)

            # Save immediately for chaining
            A_vals.append(result["A"])
            b_vals.append(result["b"])
            mu_vals.append(result["mu"])

            # Update self.param so the next call can use the last fitted values
            self.param["fit"][dist]["A"] = np.array(A_vals)
            self.param["fit"][dist]["b"] = np.array(b_vals)
            self.param["fit"][dist]["mu"] = np.array(mu_vals)

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
        A = self.fitting_functions["default"]["A"](log_axion_mass)
        b = self.fitting_functions["default"]["b"](log_axion_mass)
        mu = self.fitting_functions["default"]["mu"](log_axion_mass)

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

    ####################################################################################################################
    ########################################### FIND PARAMETER MASS DEPENDENCE #########################################
    ####################################################################################################################
    def compute_functional_fit(self, dist="f(q)_q2"):
        """
        Fit A(m_a), b(m_a), and μ(m_a) as functions of log(axion mass).
        Stores the new fit functions locally and pushes them into the process.
        """
        new_fits = {}
        log_masses = np.log(self.param["input"]["ma_arr"])

        # --- Fit A(m_a) ---
        try:
            result = curve_fit(
                self.fitting_functions["default"]["A"],
                log_masses,
                self.param["fit"][dist]["A"],
                maxfev=90000
            )
            popt_A, _ = result
            new_fits["A"] = lambda x: self.fitting_functions["default"]["A"](x, *popt_A)
            print(f"Updated fit parameters for A: {popt_A}")
        except Exception as e:
            print(f"Error fitting parameter A: {e}")
            new_fits["A"] = None

        # --- Fit b(m_a) ---
        try:
            result = curve_fit(
                self.fitting_functions["default"]["b"],
                log_masses,
                self.param["fit"][dist]["b"],
                maxfev=90000
            )
            popt_b, _ = result
            new_fits["b"] = lambda x: self.fitting_functions["default"]["b"](x, *popt_b)
            print(f"Updated fit parameters for b: {popt_b}")
        except Exception as e:
            print(f"Error fitting parameter b: {e}")
            new_fits["b"] = None

        # --- Fit μ(m_a) ---
        try:
            result = curve_fit(
                self.fitting_functions["default"]["mu"],
                log_masses,
                self.param["fit"][dist]["mu"],
                maxfev=90000
            )
            popt_mu, _ = result
            new_fits["mu"] = lambda x: self.fitting_functions["default"]["mu"](x, *popt_mu)
            print(f"Updated fit parameters for μ: {popt_mu}")
        except Exception as e:
            print(f"Error fitting parameter μ: {e}")
            new_fits["mu"] = None

        # --- Store locally ---
        self.fitting_functions["test_fit"] = new_fits

    def compute_polynomial_fit(self, dist="f(q)_q2", poly_order=13):
        """
        Fit A(m_a), b(m_a), and μ(m_a) as polynomials of log(axion mass).
        Stores the new fits locally and pushes them into the process.
        """
        new_fits_poly = {}
        log_masses = np.log(self.param["input"]["ma_arr"])

        try:
            coeffs_A = np.polyfit(log_masses, self.param["fit"][dist]["A"], poly_order)
            new_fits_poly["A"] = np.poly1d(coeffs_A)
            print(f"Polynomial fit coefficients for A: {coeffs_A}")
        except Exception as e:
            print(f"Error fitting polynomial for A: {e}")
            new_fits_poly["A"] = None

        try:
            coeffs_b = np.polyfit(log_masses, self.param["fit"][dist]["b"], poly_order)
            new_fits_poly["b"] = np.poly1d(coeffs_b)
            print(f"Polynomial fit coefficients for b: {coeffs_b}")
        except Exception as e:
            print(f"Error fitting polynomial for b: {e}")
            new_fits_poly["b"] = None

        try:
            coeffs_mu = np.polyfit(log_masses, self.param["fit"][dist]["mu"], poly_order)
            new_fits_poly["mu"] = np.poly1d(coeffs_mu)
            print(f"Polynomial fit coefficients for μ: {coeffs_mu}")
        except Exception as e:
            print(f"Error fitting polynomial for μ: {e}")
            new_fits_poly["mu"] = None

        # --- Store locally ---
        self.fitting_functions["test_polynomial"] = new_fits_poly

    # ----------------------------------------- EVALUATE PARAMETER DEPENDENCE ---------------------------------------- #
    def get_fitted_parameter_value(self, param_type, x, fit_type="default"):
        """
        Evaluate a stored or process fitting function at log(m_a).
        """
        if fit_type not in self.fitting_functions:
            raise ValueError(f"Invalid fit type '{fit_type}'. Options: {list(self.fitting_functions.keys())}")

        # Case: "default" or "best" should always reflect process state
        if fit_type in ("default", "best"):
            return self.process.evaluate_parameter(param_type, x)

        # Case: stored fit
        func = self.fitting_functions[fit_type].get(param_type)
        if func is None:
            raise ValueError(f"No function for {param_type} in fit '{fit_type}'")
        return func(x)

    ####################################################################################################################
    ########################################### CHANGE CLASS SETTINGS AND GET DATA #####################################
    ####################################################################################################################
    # ----------------------------------------- CLASS SETTINGS ------------------------------------------------------- #
    def switch_fitting_function(self, param_type, function_name):
        """
        Switch fitting function for a parameter, delegating to process if needed.
        """
        # Case 1: Stored fit (e.g., "test_polynomial")
        if function_name in self.fitting_functions and isinstance(self.fitting_functions[function_name], dict):
            if param_type not in self.fitting_functions[function_name]:
                raise ValueError(f"Stored fit '{function_name}' has no param '{param_type}'.")

            self.fitting_functions["default"][param_type] = self.fitting_functions[function_name][param_type]
            print(f"Switched {param_type} fitting function to stored fit '{function_name}'.")
            return

        # Case 2: Delegate to process module
        self.process.set_fitting_function(param_type, function_name)

        # Refresh default wrappers
        self.fitting_functions["default"] = self._wrap_process_defaults()
        print(f"Switched {param_type} fitting function in process to '{function_name}'.")

    def show_available_functions(self):
        """Prints all available fitting function choices for each parameter (A, b, and mu)."""
        print("\n Available Fitting Functions:")
        for param_type, functions in self.process.function_candidates.items():
            print(f"  - {param_type}: {list(functions.keys())}")

    # ----------------------------------------- GET DATA ------------------------------------------------------------- #
    def get_fit_parameters(self, dist="f(q)_q2"):
        """return all fit parameters for all distributions in class attribute"""
        return self.param["fit"][dist]

    def get_interpolated_function(self, dist, parameter):
        """
        Creates and returns an interpolation function for a given parameter and distribution.

        Parameters:
        - dist (str): Distribution type, e.g., "f(q)", "f(q)_q1", "f(q)_q2", "f(q)_q3".
        - parameter (str): Parameter to interpolate, e.g., "A", "b", "mu".

        Returns:
        - Callable function that interpolates the parameter based on log(m_a).
        """
        # Ensure the given distribution and parameter exist
        if dist not in self.param["fit"]:
            raise ValueError(f"Invalid distribution type: {dist}")

        if parameter not in self.param["fit"][dist]:
            raise ValueError(f"Invalid parameter: {parameter}")

        # Extract axion masses (m_a) and corresponding parameter values
        ma_arr = self.param["input"]["ma_arr"]  # m_a values in eV
        param_values = self.param["fit"][dist][parameter]  # Corresponding parameter values

        # Check if data is available for interpolation
        if len(ma_arr) == 0 or len(param_values) == 0:
            raise ValueError(f"No data available for interpolation of {parameter} in {dist}.")

        # Compute log(m_a)
        log_ma = np.log(ma_arr)

        # Create and return the interpolation function dynamically
        return interp1d(log_ma, param_values, kind='slinear', fill_value='extrapolate')
