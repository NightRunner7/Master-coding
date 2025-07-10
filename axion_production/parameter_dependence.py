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
from axion_production import get_process
from axion_production.distribution_first_interpolation import FirstInterpolation

class ParametersAxionMassDependence(FirstInterpolation):
    """
    A class for analyzing the dependence of axion distribution parameters on axion mass.

    Functionality:
    - Loads best-fit parameters from a selected production process.
    - Interpolates extracted data and fits functional dependencies.
    - Provides access to fitted functions for A(m_a), b(m_a), and μ(m_a).
    """

    # ----------------------------------------- INITIALIZATION ------------------------------------------------------ #
    def __init__(self, process_name, file_path):
        """
        Initialize the class with a specific axion production process.

        Parameters:
        - process_name: The selected axion production process module (e.g., muon_scattering).
        - file_path (str): Path to the input data file.
        """
        # --- Inherit from FirstInterpolation (load data and perform first interpolate of distributions) ---
        super().__init__(file_path)

        # --- Load production process dynamically ---
        self.process = get_process(process_name)

        # --- Store available fitting functions ---
        self.fitting_functions = dict()

        # Best available fit (from external sources)
        self.fitting_functions["best"] = {
            param: lambda x, param=param: self.process.DEFAULT_FITTING_FUNCTIONS[param](x, *self.process.DEFAULT_COEFFS[param])
            for param in ["A", "b", "mu"]
        }

        # Default functions (used in standard computations)
        # self.fitting_functions["default"] = self.process.DEFAULT_FITTING_FUNCTIONS.copy()
        self.fitting_functions["default"] = {
            param: lambda x, param=param: self.process.DEFAULT_FITTING_FUNCTIONS[param](x, *self.process.DEFAULT_COEFFS[param])
            for param in ["A", "b", "mu"]
        }

        # Additional fits (to be computed later)
        self.fitting_functions["test_polynomial"] = None  # Placeholder for polynomial fit
        self.fitting_functions["test_fit"] = None  # Placeholder for test fit

        # --- Initialize dictionary for fitted parameters ---
        self.param["fit"] = {
            dist: {"A": None, "b": None, "mu": None}
            for dist in ["f(q)", "f(q)_q1", "f(q)_q2", "f(q)_q3"]
        }

        # Fit range settings
        self.param["fit"]["q_min"] = 0.01  # 0.001
        self.param["fit"]["q_max"] = 19.99  # 19.999
        self.param["fit"]["q_step"] = 0.01  # 0.001

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

        # Perform the fit
        result: tuple = curve_fit(model_functions[dist], q_values, data_to_fit, method='trf')
        popt, pcov = result  # This should now suppress PyCharm's warning
        # popt, _ = curve_fit(model_functions[dist], q_values, data_to_fit, method='lm', maxfev=50000)

        return {"A": popt[0], "b": popt[1], "mu": popt[2]}

    def fit_all_distributions(self, dist="f(q)_q2"):
        """Fit all distributions in the dataset and store results."""
        results = [self.fit_single_distribution(i, dist) for i in range(self.distributions["input"]["N"])]
        for param in ["A", "b", "mu"]:
            self.param["fit"][dist][param] = np.array([res[param] for res in results])

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
    # ----------------------------------------- COMPUTING BEST FITS -------------------------------------------------- #
    def compute_functional_fit(self, dist="f(q)_q2"):
        """
        Fit A(m_a), b(m_a), and μ(m_a) as functions of log(axion mass).
        Stores the new fit functions in `self.new_fit_parameters`.

        Parameters:
        - dist (str): Specifies which distribution's fitted parameters should be used.
        """
        new_fits = {}

        # Get log of axion masses
        log_masses = np.log(self.param["input"]["ma_arr"])

        # --- Fit A(m_a) ---
        try:
            # Set initial guess to ones (or some reasonable default values)
            initial_guess = [
                1.25540634, 25.23140449, -213.21252559, -1.50216418, -176.84370719, -40.06637671, 245.14623916,
                21.13818352, 16.5282045, -186.7395364, -33.52252315, -161.47819936, -8.70319333,  219.62748638,
                18.73939703, - 2.
            ]
            # initial_guess = [1.04720071] + [1] * 14 + [-2.0]

            # Perform the fit
            result: tuple = curve_fit(self.fitting_functions["default"]["A"],
                                      log_masses,
                                      self.param["fit"][dist]["A"],
                                      p0=initial_guess,
                                      maxfev=90000)
            popt_A, pcov_A = result
            new_fits["A"] = lambda x: self.fitting_functions["default"]["A"](x, *popt_A)
            print(f"Updated fit parameters for A: {popt_A}")
        except Exception as e:
            print(f"Error fitting parameter A: {e}")
            new_fits["A"] = None

        # --- Fit b(m_a) ---
        try:
            # Set initial guess to ones (or some reasonable default values)
            initial_guess = [
                -2.77576175e+00, 1.82940840e+00, 2.22230261e+00,  1.63986335e+00, -6.42279540e-02, -2.37363679e-04,
                -1.46742969e-03, -1.62412865e-05, 1.48303556e-02,  1.50951749e-02, -6.37351025e-02, -8.46885428e-02,
                1.77349302e-01,  2.40159283e-01, -6.27056763e-01,  3.90298915e-01, -1.09621190e-01
            ]
            # initial_guess = [-2.77576175, 1.82940843, 2.02864447, 1.63986335, -0.06619925346515682] + [1] * 12

            # Perform the fit
            result: tuple = curve_fit(
                self.fitting_functions["default"]["b"],  # The function to fit
                log_masses,  # X-data: log(m_a)
                self.param["fit"][dist]["b"],  # Y-data: parameter values to fit
                p0=initial_guess,
                maxfev=90000)
            popt_b, pcov_b = result
            new_fits["b"] = lambda x: self.fitting_functions["default"]["b"](x, *popt_b)
            print(f"Updated fit parameters for b: {popt_b}")
        except Exception as e:
            print(f"Error fitting parameter b: {e}")
            new_fits["b"] = None

        # --- Fit μ(m_a) ---
        try:
            # Set initial guess to ones (or some reasonable default values)
            initial_guess = [-0.89232017,  0.49383594, -2.12668891]
            # initial_guess = [0.3, 0.5, 0.1]

            # Perform the fit
            result: tuple = curve_fit(self.fitting_functions["default"]["mu"],
                                      log_masses, self.param["fit"][dist]["mu"],
                                      p0=initial_guess,
                                      maxfev=90000)
            popt_mu, pcov_mu = result
            new_fits["mu"] = lambda x: self.fitting_functions["default"]["mu"](x, *popt_mu)
            print(f"Updated fit parameters for μ: {popt_mu}")
        except Exception as e:
            print(f"Error fitting parameter μ: {e}")
            new_fits["mu"] = None

        # Store in class attribute
        self.fitting_functions["test_fit"] = new_fits

    def compute_polynomial_fit(self, dist="f(q)_q2", poly_order=13):
        """
        Fit A(m_a), b(m_a), and μ(m_a) as polynomials of log(axion mass).
        Stores the new polynomial fit functions in `self.new_fit_parameters_poly`.

        Parameters:
        - dist (str): Specifies which distribution's fitted parameters should be used.
        - poly_order (int): Degree of the polynomial fit.
        """
        new_fits_poly = {}

        # Get log of axion masses
        log_masses = np.log(self.param["input"]["ma_arr"])

        # --- Fit A(m_a) with polynomial ---
        try:
            poly_coeffs_A = np.polyfit(log_masses, self.param["fit"][dist]["A"], poly_order)
            new_fits_poly["A"] = np.poly1d(poly_coeffs_A)
            print(f"Polynomial fit coefficients for A: {poly_coeffs_A}")
        except Exception as e:
            print(f"Error fitting polynomial for A: {e}")
            new_fits_poly["A"] = None

        # --- Fit b(m_a) with polynomial ---
        try:
            poly_coeffs_b = np.polyfit(log_masses, self.param["fit"][dist]["b"], poly_order)
            new_fits_poly["b"] = np.poly1d(poly_coeffs_b)
            print(f"Polynomial fit coefficients for b: {poly_coeffs_b}")
        except Exception as e:
            print(f"Error fitting polynomial for b: {e}")
            new_fits_poly["b"] = None

        # --- Fit μ(m_a) with polynomial ---
        try:
            poly_coeffs_mu = np.polyfit(log_masses, self.param["fit"][dist]["mu"], poly_order)
            new_fits_poly["mu"] = np.poly1d(poly_coeffs_mu)
            print(f"Polynomial fit coefficients for μ: {poly_coeffs_mu}")
        except Exception as e:
            print(f"Error fitting polynomial for μ: {e}")
            new_fits_poly["mu"] = None

        # Store in class attribute
        self.fitting_functions["test_polynomial"] = new_fits_poly

    # ----------------------------------------- EVALUATE PARAMETER DEPENDENCE ---------------------------------------- #
    def get_fitted_parameter_value(self, param_type, x, fit_type="default"):
        """
        Evaluate a stored fitting function at a given x (log(m_a)).

        Parameters:
        - param_type (str): The parameter to evaluate ("A", "b", or "mu").
        - x (float or np.ndarray): The input value (log of axion mass).
        - fit_type (str): The category of fitting function to use (default: "default").

        Returns:
        - float or np.ndarray: Evaluated function output.
        """
        # Validate fit type first
        if fit_type not in self.fitting_functions:
            available_types = list(self.fitting_functions.keys())
            raise ValueError(f"Invalid fit category '{fit_type}'. Available options: {available_types}")

        # Validate parameter type
        fit_functions = self.fitting_functions[fit_type]
        if param_type not in fit_functions:
            available_params = list(fit_functions.keys())
            raise ValueError(f"Invalid parameter '{param_type}'. Available options in '{fit_type}': {available_params}")

        # Get and apply the function
        return fit_functions[param_type](x)

    ####################################################################################################################
    ########################################### CHANGE CLASS SETTINGS AND GET DATA #####################################
    ####################################################################################################################
    # ----------------------------------------- CLASS SETTINGS ------------------------------------------------------- #
    def switch_fitting_function(self, param_type, function_name):
        """
        Set a fitting function for a specific parameter in the 'default' category.

        Parameters:
            param_type (str): The parameter to modify ("A", "b", or "mu").
            function_name (str): Either a predefined function name (e.g., "rational_6th")
                                 or a stored fit (e.g., "test_polynomial").

        Example:
        ```
        obj.switch_fitting_function("A", "rational_6th")  # Use predefined function
        obj.switch_fitting_function("A", "test_polynomial")    # Use stored polynomial fit
        ```
        """
        # --- Validate parameter type ---
        if param_type not in self.fitting_functions["default"]:
            raise ValueError(
                f"Invalid parameter type: '{param_type}'. Available: {list(self.fitting_functions['default'].keys())}."
            )

        # --- Case 1: Switch to a stored fit (e.g., "test_polynomial") ---
        if function_name in self.fitting_functions and isinstance(self.fitting_functions[function_name], dict):
            if param_type not in self.fitting_functions[function_name]:
                raise ValueError(
                    f"The selected stored fit '{function_name}' does not contain parameter '{param_type}'.")

            # Update the default fitting function
            self.fitting_functions["default"][param_type] = self.fitting_functions[function_name][param_type]
            print(f"Switched {param_type} fitting function to stored fit '{function_name}'.")
            return

        # --- Case 2: Switch to a predefined function ---
        if param_type not in self.process.function_candidates:
            raise ValueError(
                f"Invalid parameter type: '{param_type}'. Available: {list(self.process.function_candidates.keys())}")

        available_functions = self.process.function_candidates[param_type]
        if function_name not in available_functions:
            raise ValueError(
                f"Invalid function '{function_name}' for {param_type}. Available options: {list(available_functions.keys())}"
            )

        # --- Update to the predefined function ---
        self.fitting_functions["default"][param_type] = available_functions[function_name]
        print(f"Switched {param_type} fitting function to: {function_name}.")

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


# ####################################### CROSS CHECKS ############################################################### #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Set The Data ---
    # selected_file_path = '../Maxim-data/ma_distributions_mu_scattering.dat'  # Set the correct file path
    # parameterDependence = ParametersAxionMassDependence("muon_scattering", selected_file_path)
    # selected_file_path = '../Maxim-data/Distributions_fa_tau_dec.dat'  # Set the correct file path
    selected_file_path = '../Maxim-data/Distributions_fa_tau_dec.dat'  # Set the correct file path
    parameterDependence = ParametersAxionMassDependence("taon_decay", selected_file_path)


    # --- DISTRIBUTION: f(q)*q^3 ---
    distribution = "f(q)_q2"  # "f(q)_q2"
    parameterDependence.fit_all_distributions(dist=distribution)

    # Set bounds for log(m_a)

    log_m_start = -5.20  # -3.00 (muon), -3.00 (electron), -5.2 (taon decays)
    log_m_end = 2.00     #  2.00 (muon),  2.00 (electron),  2.0 (taon decays)
    log_m_arr_set = np.linspace(log_m_start, log_m_end, 500)  # Log(m_a) values for testing

    # --- Interpolation ---
    interpolated_A_value = parameterDependence.get_interpolated_function(dist=distribution, parameter="A")(log_m_arr_set)
    interpolated_b_value = parameterDependence.get_interpolated_function(dist=distribution, parameter="b")(log_m_arr_set)
    interpolated_mu_value = parameterDependence.get_interpolated_function(dist=distribution, parameter="mu")(log_m_arr_set)

    # --- Base Data ---
    m_arr_input = parameterDependence.get_axion_masses()
    log_m_arr_input = np.log(m_arr_input)
    fit_parameter = parameterDependence.get_fit_parameters(dist=distribution)

    # --- Function Fitting: A(log(m_a)), b(log(m_a)), and μ(log(m_a)) ---
    # --- polynomial
    parameterDependence.compute_polynomial_fit(dist=distribution)

    # --- function fit
    # parameterDependence.switch_fitting_function("A", "piecewise")
    # parameterDependence.switch_fitting_function("b", "b_function")
    # parameterDependence.switch_fitting_function("mu", "exp_decay")
    # parameterDependence.compute_functional_fit(dist=distribution)

    # --- PLOT EACH PARAMETER SEPARATELY ---
    name_of_parameters = ["A", "b", "mu"]
    interpolated_param = [interpolated_A_value, interpolated_b_value, interpolated_mu_value]
    points_param = [fit_parameter['A'], fit_parameter['b'], fit_parameter['mu']]
    titles = [r"$A(\log(m_{a}))$", r"$b(\log(m_{a}))$", r"$-\mu(\log(m_{a}))$"]

    colors = ['blue', 'green', 'purple']  # Different colors for better distinction

    for i in range(3):
        plt.figure(figsize=(10, 10))
        plt.title(f"{titles[i]}", fontsize=16)

        # Scatter plot of extracted parameter values
        if name_of_parameters[i] == "mu":
            plt.scatter(log_m_arr_input, (-1)*points_param[i],
                        color=colors[i], marker='o', edgecolors='black',
                        alpha=0.8, label='Extracted Parameters')
        else:
            plt.scatter(log_m_arr_input, points_param[i],
                        color=colors[i], marker='o', edgecolors='black',
                        alpha=0.8, label='Extracted Parameters')


        # Plot interpolation of extracted values
        if name_of_parameters[i] == "mu":
            plt.plot(log_m_arr_set, (-1)*interpolated_param[i],
                     linestyle='--', linewidth=2, color='black',
                     label='Interpolated Data')
        else:
            plt.plot(log_m_arr_set, interpolated_param[i],
                     linestyle='--', linewidth=2, color='black',
                     label='Interpolated Data')


        # Plot fitted function for the parameter
        if name_of_parameters[i] == "mu":
            plt.plot(log_m_arr_set, (-1)*parameterDependence.get_fitted_parameter_value(name_of_parameters[i],
                                                                                        log_m_arr_set,
                                                                                      "test_polynomial"),  # test_polynomial, test_fit
                     linestyle='-.', linewidth=2, color='red',
                     label='Fit Function')
        else:
            plt.plot(log_m_arr_set, parameterDependence.get_fitted_parameter_value(name_of_parameters[i],
                                                                                   log_m_arr_set,
                                                                                 "test_polynomial"),  # test_polynomial, test_fit
                     linestyle='-.', linewidth=2, color='red',
                     label='Fit Function')

        # Labels and formatting
        plt.xlabel(r'$\log(m_{a})$', fontsize=12)
        plt.ylabel(titles[i], fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)

        # Set log scale only for the y-axis of μ(m_a)
        # if name_of_parameters[i] == "mu":
        #     plt.yscale("log")

        # Show each plot separately
        plt.show()


    for i in range(0, len(m_arr_input)):
        ma = m_arr_input[i]
        val_parameter_A = parameterDependence.get_fitted_parameter_value("A", np.log(ma), "test_polynomial")  # test_polynomial, test_fit
        val_parameter_b = parameterDependence.get_fitted_parameter_value("b", np.log(ma), "test_polynomial")
        val_parameter_mu = parameterDependence.get_fitted_parameter_value("mu", np.log(ma), "test_polynomial")

        print("--------------------------------------------------------------------------------------------------------")
        print(f"input A: {fit_parameter['A'][i]}, fitted A: {val_parameter_A}")
        print(f"input b: {fit_parameter['b'][i]}, fitted b: {val_parameter_b}")
        print(f"input mu: {fit_parameter['mu'][i]}, fitted mu: {val_parameter_mu}")
