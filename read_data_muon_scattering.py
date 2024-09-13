import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

class DistributionOfMuonScattering:
    def __init__(self, file_path):
        """
        :param file_path:
        """
        self.file_path = file_path
        self.q_arr = None  # co-moving momenta, [dimensionless]
        self.ma_arr = None  # axion mass, [eV]
        self.fa_arr = None  # decay constant, [GeV]
        self.distribution_function = None  # f(q), [dimensionless]
        self.distributions_weighted_q_squared = None  # f(q)*q^2, [dimensionless]
        self.distributions_weighted_q_cubed = None  # f(q)*q^3, [dimensionless]
        self.num_distributions = None

        # --- We have to load data and update class
        self.load_data()
        self.compute_distributions()
        self.fa_arr = self.calculate_fa(self.ma_arr)

        # --- Do a interpolation of stored data
        self.interpolation_dist_functions = None
        self.transpose_and_interpolate()

        # --- Take care about fit distribution
        self.q_min = 0.001
        self.q_max = 19.8
        self.q_step = 0.005  # q_step > q_min
        self.fit_parameters = {
            'A': np.array([]),
            'b': np.array([]),
            'mu': np.array([])
        }
        self.polynomial_parameters_fit = None

    # ----------------------------------------- BASE METHODS --------------------------------------------------------- #
    @staticmethod
    def log_space(start, end, n):
        """Generate logarithmically spaced points."""
        return np.exp(np.linspace(np.log(start), np.log(end), n))

    @staticmethod
    def calculate_fa(ma):
        """
        Calculate the f_a value from a given axion mass m_a in eV based on the axion mass
        formula from particle physics. The relationship between the axion decay constant f_a
        and the axion mass m_a is given by the specific scale factor derived from theoretical
        models which predict f_a inversely proportional to m_a.

        This specific formula is referenced from a particle physics review publication:
        https://pdg.lbl.gov/2023/reviews/rpp2023-rev-axions.pdf

        Parameters:
        ma (float): The mass of the axion in eV.

        Returns:
        float: The axion decay constant f_a in GeV.
        """
        fa = (5.691 * 10 ** 6) / ma  # [GeV]
        return fa  # [GeV]

    @staticmethod
    def calculate_ma(fa):
        """
        Calculate the axion mass m_a from a given axion decay constant f_a in GeV based on the inverse
        of the axion mass formula from particle physics. This relationship is derived from theoretical
        models which predict that the axion mass is inversely proportional to the decay constant.

        This method computes the mass using a standard equation provided in the same reference as before:
        https://pdg.lbl.gov/2023/reviews/rpp2023-rev-axions.pdf

        Parameters:
        fa (float): The decay constant of the axion in GeV.

        Returns:
        float: The axion mass m_a in eV.
        """
        ma = (5.691 * 10 ** 6) / fa  # [eV]
        return ma  # [eV]

    def load_data(self):
        """Load and parse data from the CSV file, assuming a comment line at the start."""
        data = pd.read_csv(self.file_path, comment='#', header=None)
        self.ma_arr = 10 ** 9 * data.iloc[:, 0].values  # [eV]
        self.distributions_weighted_q_squared = data.iloc[:, 1:].values  # f(q) * q^2, [dimensionless]

    def compute_distributions(self):
        """Compute the distributions using the loaded data."""
        # Generate 200 points from 1e-4 to 20.0 for qValuesList
        self.q_arr = self.log_space(1e-4, 20.0, 200)  # [dimensionless]

        # Calculating the distributions
        self.distribution_function = self.distributions_weighted_q_squared / self.q_arr[None, :] ** 2
        self.distributions_weighted_q_cubed = self.distributions_weighted_q_squared * self.q_arr[None, :]

        # Set the number of distributions
        self.num_distributions = len(self.ma_arr)

    # --- INTERPOLATION OF LOADED DATA
    def transpose_and_interpolate(self):
        """
        Transposes the distribution data and interpolates it to create a continuous
        function for each distribution using cubic spline interpolation.

        This method modifies the class by adding a list of interpolation functions
        that can be evaluated at any q-value within the observed range.

        Attributes:
        ax_dist_functions (list): List of interpolation functions for each distribution.
        """
        # select the base distribution
        base_distribution = self.distributions_weighted_q_cubed

        # Transpose data: create pairs (q, distribution) for each distribution list
        transposed_data = [np.column_stack((self.q_arr, base_distribution[i]))
                           for i in range(self.num_distributions)]

        # Interpolate data using cubic splines
        self.interpolation_dist_functions = [interp1d(data[:, 0], data[:, 1], kind='cubic')
                                             for data in transposed_data]

    # ----------------------------------------- APPROX DISTRIBUTION -------------------------------------------------- #
    @staticmethod
    def f_approx(q, A, b, mu):
        """
        Approximate function for fitting to the f(q)*q^3 distribution in terms of the
        co-moving momenta q.

        Parameters:
        q (float or np.ndarray): The co-moving momenta, which can be a single float or a numpy array.
        A (float): Parameter A in the exponential term.
        b (float): Parameter b in the exponential term.
        mu (float): Parameter mu, added inside the inverse term.

        Returns:
        np.ndarray: The evaluated function values at each point q, according to the given parameters.
        """
        return q ** 2 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** -1

    # ----------------------------------------- FIT DISTRIBUTION ----------------------------------------------------- #
    def find_params_fit_distribution(self, index, q_min, q_max, q_step):
        """
        Fits the approximation function to a selected interpolated distribution
        using non-linear least squares optimization.

        This method evaluates the interpolated distribution function at a range of q values,
        fits the approximation model to these data points, and returns the best-fit parameters.

        Parameters:
        index (int): The index of the distribution to fit from the list of distributions.
        q_min (float): The minimum value of q to use for fitting.
        q_max (float): The maximum value of q to use for fitting.
        q_step (float): The step size to increment q values within the fitting range.

        Returns:
        dict: A dictionary containing the best-fit parameters 'A', 'b', and 'mu'.
        """
        if index >= self.num_distributions:
            print("Index out of range. Please provide a valid distribution index.")
            return

        # Generate q values
        q_values = np.arange(q_min, q_max + q_step, q_step)
        # Evaluate the interpolated function at these q values
        interp_func = self.interpolation_dist_functions[index]
        data_to_fit = interp_func(q_values)

        # Define a local function for curve fitting if necessary
        def model(q, A, b, mu):
            return self.f_approx(q, A, b, mu)

        # Perform the fit
        popt, pcov = curve_fit(model, q_values, data_to_fit, method='trf')

        # Extract best-fit parameters
        best_fit_params = {'A': popt[0], 'b': popt[1], 'mu': popt[2]}

        return best_fit_params

    def find_all_fit_distribution_parameters(self):
        """Calculates fit parameters for all distributions and stores results in class attribute."""
        for index in range(0, self.num_distributions):
            # calculate the fit parameters
            best_fit_params = self.find_params_fit_distribution(index, self.q_min, self.q_max, self.q_step)

            if best_fit_params:  # Ensure that best_fit_params is not None
                # appending this parameters
                self.fit_parameters['A'] = np.append(self.fit_parameters['A'], best_fit_params['A'])
                self.fit_parameters['b'] = np.append(self.fit_parameters['b'], best_fit_params['b'])
                self.fit_parameters['mu'] = np.append(self.fit_parameters['mu'], best_fit_params['mu'])
                # check how fast these parameters can be found
                print(f"Fit distribution parameters found for distribution: {index}")
            else:
                print(f"Failed to find fit parameters for distribution: {index}")

    def find_polynomial_fit_of_parameters(self):
        """
        Fits polynomials to the parameters A, b, and mu as functions of the log of the axion masses.
        Stores the polynomial functions for each parameter.
        """
        # Set up polynomial
        self.polynomial_parameters_fit = {}

        # Get the log_m_a
        log_masses = np.log(self.return_axion_masses())

        # deal with parameter A
        coefficients = np.polyfit(log_masses, self.fit_parameters['A'], 12)
        self.polynomial_parameters_fit['A'] = np.poly1d(coefficients)
        # deal with parameter b
        coefficients = np.polyfit(log_masses, self.fit_parameters['b'], 11)
        self.polynomial_parameters_fit['b'] = np.poly1d(coefficients)
        # deal with parameter mu
        coefficients = np.polyfit(log_masses, self.fit_parameters['mu'], 12)
        self.polynomial_parameters_fit['mu'] = np.poly1d(coefficients)

    # ----------------------------------------- PRINT DATA ----------------------------------------------------------- #
    def print_check_of_load_data(self):
        """Print summary statistics and information about the distributions."""
        print("q_arr", self.q_arr)
        print("ma_arr", self.ma_arr)
        print("fa_arr", self.fa_arr)
        print("Number of Distributions:", self.num_distributions)
        if self.num_distributions > 0:
            print("First 5 maValues:", self.ma_arr[:5])
            print("First row of distributions:", self.distribution_function[0][:5])

    # ----------------------------------------- SANITY CHECK --------------------------------------------------------- #
    def plot_interpolation_check(self, index):
        """
        Plots the original data points and the interpolated function for a specified distribution index.

        Parameters:
        index (int): Index of the distribution to plot (e.g., 199 for the 200th distribution).
        """
        if index >= self.num_distributions:
            print("Index out of range. Please provide a valid distribution index.")
            return

        # Extract original data points
        q_values = self.q_arr
        original_distribution = self.distributions_weighted_q_cubed[index]

        # Evaluate the interpolated function at these q values
        q_fine = np.linspace(min(q_values), max(q_values), 800)
        interpolated_values = self.interpolation_dist_functions[index](q_fine)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(q_fine, interpolated_values, label='Interpolated Curve', color='blue')
        plt.scatter(q_values, original_distribution, color='red', s=20, label='Original Data Points')
        plt.title(f'Sanity Check for Distribution Index: {index}')
        plt.xlabel('q')
        plt.ylabel('f(q) * q^3')
        plt.legend()
        plt.grid(True)
        plt.show()

    def fit_distribution_check(self, index, xmin, xmax, step):
        """
        Fit the approximation function to the interpolated distribution.

        Parameters:
        index (int): Index of the distribution to fit.
        xmin (float): Minimum q value for fitting.
        xmax (float): Maximum q value for fitting.
        step (float): Step size to generate q values for fitting.

        Returns:
        dict: Dictionary containing best fit parameters.
        """
        if index >= self.num_distributions:
            print("Index out of range. Please provide a valid distribution index.")
            return

        # Generate q values
        q_values = np.arange(xmin, xmax + step, step)
        # Evaluate the interpolated function at these q values
        interp_func = self.interpolation_dist_functions[index]
        data_to_fit = interp_func(q_values)

        # Define a local function for curve fitting if necessary
        def model(q, A, b, mu):
            return self.f_approx(q, A, b, mu)

        # Perform the fit
        popt, pcov = curve_fit(model, q_values, data_to_fit, method='trf')

        # Extract best-fit parameters
        best_fit_params = {'A': popt[0], 'b': popt[1], 'mu': popt[2]}

        # Optionally: Plot the fit against the data
        plt.figure(figsize=(10, 5))
        plt.plot(q_values, data_to_fit, '--', label='Data')
        plt.plot(q_values, model(q_values, *popt), '-', label='Fit')
        plt.xlabel('q')
        plt.ylabel('f(q) * q^3')
        plt.title('Fit to Interpolated Distribution')
        plt.legend()
        plt.show()

        return best_fit_params

    # ----------------------------------------- RETURN DATA ---------------------------------------------------------- #
    def return_axion_masses(self):
        """return all axion masses in class attribute"""
        return self.ma_arr

    def return_comoving_momenta(self):
        """return all co-moving momenta in class attribute"""
        return self.q_arr

    def return_fit_parameters(self):
        """return all fit parameters for all distributions in class attribute"""
        return self.fit_parameters

    def return_polynomial_of_parameters(self):
        """return polynomial fit to all parameters of approximation of distribution"""
        return self.polynomial_parameters_fit


# Usage
file_path = './Maxim-muon-scattering/ma_distributions_mu_scattering.dat'  # Set the correct file path
analyzer = DistributionOfMuonScattering(file_path)
#
# # --- Sanity Checks
# Plot for specific distribution index, e.g., 199
# analyzer.plot_interpolation_check(199)  # Remember to adjust the index based on your actual data count
# analyzer.print_check_of_load_data()
#
# # Fit to the first distribution (index 0) from q = 0.0 to q = 11 with a step of 0.005
# # best_fit_params = analyzer.fit_distribution_check(199, 0.001, 19.8, 0.005)
# # print("Best fit parameters:", best_fit_params)
#
# # --- plot of parameters how they evolve
# analyzer.find_all_fit_distribution_parameters()  # find all parameters
# # return data from the class
# axion_masses = analyzer.return_axion_masses()
# log_masses = np.log(axion_masses)
# fit_parameters = analyzer.return_fit_parameters()

# # Create the plot: A
# plt.figure(figsize=(10, 6))
# plt.scatter(log_masses, fit_parameters['A'], color='black', label=f'Parameter: {"A"}')
# plt.xlabel('log(m_a) [eV]')
# plt.ylabel(f'parameter: {"A"}(m_a)')
# plt.title(f'Parameter {"A"} vs. log(m_a)')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# # Create the plot: b
# plt.figure(figsize=(10, 6))
# plt.scatter(log_masses, fit_parameters['b'], color='black', label=f'Parameter: {"b"}')
# plt.xlabel('log(m_a) [eV]')
# plt.ylabel(f'parameter: {"b"}(m_a)')
# plt.title(f'Parameter {"b"} vs. log(m_a)')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# # Create the plot: mu
# plt.figure(figsize=(10, 6))
# plt.scatter(log_masses, fit_parameters['mu'], color='black', label=f'Parameter: {"mu"}')
# plt.xlabel('log(m_a) [eV]')
# plt.ylabel(f'parameter: {"mu"}(m_a)')
# plt.title(f'Parameter {"mu"} vs. log(m_a)')
# plt.grid(True)
# plt.legend()
# plt.show()

# # --- parameter A polynomial fit
# # Fit a polynomial of degree 2
# coefficients = np.polyfit(log_masses, fit_parameters['A'], 12)
# polynomial = np.poly1d(coefficients)
#
# # Print the polynomial equation
# print("Fitted Polynomial:", polynomial)
#
# # Plot the original data and the fitted curve
# plt.figure(figsize=(10, 5))
# plt.scatter(log_masses, fit_parameters['A'], color='black', label='Original Data')
# plt.plot(np.sort(log_masses), polynomial(np.sort(log_masses)), label='Fitted Polynomial', color='red')
# plt.xlabel('log(m_a) [eV]')
# plt.ylabel('Parameter: A(m_a)')
# plt.title('Fitting Polynomial to Parameter A vs. log(m_a)')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # --- parameter b polynomial fit
# # Fit a polynomial of degree 2
# coefficients = np.polyfit(log_masses, fit_parameters['b'], 11)
# polynomial = np.poly1d(coefficients)
#
# # Print the polynomial equation
# print("Fitted Polynomial:", polynomial)
#
# # Plot the original data and the fitted curve
# plt.figure(figsize=(10, 5))
# plt.scatter(log_masses, fit_parameters['b'], color='black', label='Original Data')
# plt.plot(np.sort(log_masses), polynomial(np.sort(log_masses)), label='Fitted Polynomial', color='red')
# plt.xlabel('log(m_a) [eV]')
# plt.ylabel('Parameter: b(m_a)')
# plt.title('Fitting Polynomial to Parameter A vs. log(m_a)')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # --- parameter mu polynomial fit
# # Fit a polynomial of degree 2
# coefficients = np.polyfit(log_masses, fit_parameters['mu'], 12)
# polynomial = np.poly1d(coefficients)
#
# # Print the polynomial equation
# print("Fitted Polynomial:", polynomial)
#
# # Plot the original data and the fitted curve
# plt.figure(figsize=(10, 5))
# plt.scatter(log_masses, fit_parameters['mu'], color='black', label='Original Data')
# plt.plot(np.sort(log_masses), polynomial(np.sort(log_masses)), label='Fitted Polynomial', color='red')
# plt.xlabel('log(m_a) [eV]')
# plt.ylabel('Parameter: mu(m_a)')
# plt.title('Fitting Polynomial to Parameter A vs. log(m_a)')
# plt.legend()
# plt.grid(True)
# plt.show()
