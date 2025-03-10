"""
Axion Distribution First Interpolation Module
=============================================

This module defines the `FirstInterpolation` class, which performs the first interpolation
of axion distribution functions extracted from a data file.

The interpolation ensures that subsequent parameter fitting steps are performed on smooth,
well-behaved distributions. The main functionalities of this module include:

1. **Data Handling:**
   - Reads axion distribution data from a file.
   - Extracts co-moving momenta (q), axion masses (m_a), and axion decay constants (f_a).

2. **Computation of Derived Distributions:**
   - Computes multiple forms of the distribution function: f(q), f(q) * q, f(q) * q², f(q) * q³.

3. **Interpolation:**
   - Uses cubic spline interpolation to construct smooth functions for f(q) and its variations.
   - Provides callable interpolation functions for further analysis.

4. **Physical Relationships:**
   - Implements the standard inverse relationship between axion mass (m_a) and decay constant (f_a).

5. **Visualization (for Debugging & Validation):**
   - Includes a plotting section that allows users to inspect the interpolation results.

Usage Example:
--------------
interpolator = FirstInterpolation(file_path="data.csv")
f_q_interp = interpolator.get_interpolated_distribution("f(q)_q2")
result = f_q_interp(some_q_value)

Author: Krzysztof Szafrański
Date: 2025-02-13
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class FirstInterpolation:
    """
    FirstInterpolation handles the first interpolation step for axion distribution functions.

    This class:
    - Loads raw axion distribution data from a file.
    - Extracts co-moving momenta (q) and axion masses (m_a).
    - Computes multiple forms of the distribution: f(q), f(q) * q, f(q) * q², f(q) * q³.
    - Uses cubic spline interpolation to create smooth representations of these functions.

    This interpolation step ensures that further analysis, such as fitting parameter dependencies,
    is based on well-behaved data.

    Attributes:
    - param (dict): Stores input parameters, including file name and q-array.
    - distributions (dict): Contains both raw and interpolated distributions.

    Usage:
    ```
    interpolator = FirstInterpolation(file_path="data.csv")
    f_q_interp = interpolator.return_interp1_dist("f(q)_q2")
    ```
    """

    def __init__(self, file_path):
        """
        Initialize FirstInterpolation and perform the first interpolation step.

        Parameters:
        - file_path (str): Path to the data file containing axion distributions.
        """
        # Store input file path
        self.param = {"input": {"file_name": file_path}}

        # Initialize placeholders for input variables
        self.param["input"]["q_arr"] = None  # Co-moving momenta [dimensionless]
        self.param["input"]["ma_arr"] = None  # Axion mass [eV]
        self.param["input"]["fa_arr"] = None  # Decay constant [GeV]

        # --- Storage for distributions ---
        self.distributions = {
            "input": {
                "f(q)": [], "f(q)_q": [], "f(q)_q2": [], "f(q)_q3": [], "N": 0
            },
            "interp1": {  # First interpolation results
                "f(q)": [], "f(q)_q": [], "f(q)_q2": [], "f(q)_q3": []
            }
        }

        # --- Load data and compute distributions ---
        self.load_data()  # Read raw data from file
        self.compute_distributions()  # Compute f(q), f(q)*q, etc.
        # self.param["input"]["fa_arr"] = self.calculate_fa(self.param["input"]["ma_arr"])
        self.param["input"]["ma_arr"] = self.calculate_ma(self.param["input"]["fa_arr"])

        # --- Perform first interpolation ---
        self.interpolate_distributions()

    # ----------------------------------------- BASE METHODS --------------------------------------------------------- #
    @staticmethod
    def generate_log_q(start, end, n):
        """Generate logarithmically spaced q values."""
        return np.logspace(np.log10(start), np.log10(end), n, base=10.0)

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

    # ----------------------------------------- FIRST INTERPOLATION -------------------------------------------------- #
    def load_data_old(self):
        """Load and parse data from the CSV file, assuming a comment line at the start."""
        data = pd.read_csv(self.param["input"]["file_name"], comment='#', header=None)
        self.param["input"]["ma_arr"] = 10 ** 9 * data.iloc[:, 0].values  # [eV]
        self.distributions["input"]["f(q)_q2"] = data.iloc[:, 1:].values  # f(q) * q^2, [dimensionless]

        # Store the number of distributions
        self.distributions["input"]["N"] = len(self.param["input"]["ma_arr"])

    def load_data(self):
        """Load and parse data from the CSV file, assuming a comment line at the start."""
        data = pd.read_csv(self.param["input"]["file_name"], comment='#', header=None)
        self.param["input"]["fa_arr"] = data.iloc[:, 0].values  # [GeV]
        self.distributions["input"]["f(q)_q2"] = data.iloc[:, 1:].values  # f(q) * q^2, [dimensionless]

        # Store the number of distributions
        self.distributions["input"]["N"] = len(self.param["input"]["fa_arr"])

    def compute_distributions(self):
        """Compute different forms of the axion distribution based on q-values."""
        self.param["input"]["q_arr"] = self.generate_log_q(1e-4, 20.0, 200)

        q: np.ndarray = self.param["input"]["q_arr"]
        f_q_q2 = self.distributions["input"]["f(q)_q2"]

        # Compute related distributions
        self.distributions["input"]["f(q)"] = f_q_q2 / (q[:, np.newaxis] ** 2).T
        self.distributions["input"]["f(q)_q1"] = f_q_q2 / (q[:, np.newaxis]).T
        self.distributions["input"]["f(q)_q3"] = f_q_q2 * (q[:, np.newaxis]).T

    # --- INTERPOLATION OF LOADED DATA
    def interpolate_distributions(self):
        """
        Perform cubic spline interpolation on f(q) * q²
        and derive interpolated functions for f(q), f(q) * q, and f(q) * q³.
        """
        q: np.ndarray = self.param["input"]["q_arr"]
        f_q_q2 = self.distributions["input"]["f(q)_q2"]

        # Interpolate f(q) * q²
        self.distributions["interp1"]["f(q)_q2"] = [
            interp1d(q, f_q_q2[i], kind='cubic', fill_value="extrapolate")
            for i in range(self.distributions["input"]["N"])
        ]

        # Define helper function for derived distributions
        def derived_interpolation(q_func, exponent):
            """
            Generate interpolated functions for derived distributions by multiplying
            the base interpolation function by q^exponent.
            """
            derived_funcs = []
            for f in q_func:
                derived_funcs.append(lambda q, func=f: func(q) * q ** exponent)
            return derived_funcs

        # Compute interpolated versions of f(q), f(q) * q, f(q) * q³
        self.distributions["interp1"]["f(q)"] = derived_interpolation(self.distributions["interp1"]["f(q)_q2"], -2)
        self.distributions["interp1"]["f(q)_q1"] = derived_interpolation(self.distributions["interp1"]["f(q)_q2"], -1)
        self.distributions["interp1"]["f(q)_q3"] = derived_interpolation(self.distributions["interp1"]["f(q)_q2"],1)

    # ----------------------------------------- GET DATA ------------------------------------------------------------- #
    def get_number_of_distributions(self):
        """Return number indicating quantity of stored distributions"""
        return self.distributions["input"]["N"]

    def get_axion_masses(self):
        """Return all stored axion masses in eV."""
        return self.param["input"]["ma_arr"]

    def get_decay_constants(self):
        """Return all stored axion decay constants in GeV."""
        return self.param["input"]["fa_arr"]

    def get_comoving_momenta(self):
        """Return all stored co-moving momenta."""
        return self.param["input"]["q_arr"]

    def get_input_distribution(self, dist="f(q)_q2"):
        """Return the input distribution for a given type."""
        return self.distributions["input"][dist]

    def get_interpolated_distribution(self, dist="f(q)_q2"):
        """Return the first interpolated distribution for a given type."""
        return self.distributions["interp1"][dist]


# ####################################### CROSS CHECKS ############################################################### #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- ALL STORED FILES ---
    filename_dist_arr = ["Distributions_fa_e_scat.dat", "Distributions_fa_mu_dec.dat", "Distributions_fa_mu_scat.dat",
                         "Distributions_fa_tau_dec.dat", "Distributions_fa_tau_scat.dat"]

    # --- Set The Data ---
    file_number = 3
    selected_file_path = f'../Maxim-data/{filename_dist_arr[file_number]}'  # Set the correct file path
    interpolator = FirstInterpolation(selected_file_path)
    index = 180
    # --- Input Data ---
    axion_mass = interpolator.get_axion_masses()[index]
    decay_constant = interpolator.calculate_fa(axion_mass)
    input_dist_q0 = interpolator.get_input_distribution(dist="f(q)")
    input_dist_q1 = interpolator.get_input_distribution(dist="f(q)_q1")
    input_dist_q2 = interpolator.get_input_distribution(dist="f(q)_q2")
    input_dist_q3 = interpolator.get_input_distribution(dist="f(q)_q3")
    input_q_arr = interpolator.get_comoving_momenta()

    # Convert to scientific notation and extract base & exponent
    axion_mass_base, axion_mass_exp = f"{axion_mass:.2e}".split("e")
    decay_constant_base, decay_constant_exp = f"{decay_constant:.2e}".split("e")
    # Format with integer exponents (ensuring proper LaTeX)
    axion_mass_formatted = rf"{float(axion_mass_base):.2f} \times 10^{{{int(axion_mass_exp)}}}"
    decay_constant_formatted = rf"{float(decay_constant_base):.2f} \times 10^{{{int(decay_constant_exp)}}}"

    # --- First Interpolation Data ---
    interp1_q_arr = np.logspace(np.log10(1e-4), np.log10(20), 500, base=10.0)
    interp1_dist_q0 = interpolator.get_interpolated_distribution(dist="f(q)")
    interp1_dist_q1 = interpolator.get_interpolated_distribution(dist="f(q)_q1")
    interp1_dist_q2 = interpolator.get_interpolated_distribution(dist="f(q)_q2")
    interp1_dist_q3 = interpolator.get_interpolated_distribution(dist="f(q)_q3")

    # --- Plotting Part ---
    # Set up figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(rf"Input Data and First Interpolation for $m_a = {axion_mass_formatted}$ eV or " 
                 rf"$f_a = {decay_constant_formatted}$ GeV", fontsize=16)

    # Define plot titles and datasets
    titles = [r'$f(q)$', r'$f(q) \cdot q$',
              r'$f(q) \cdot q^2$', r'$f(q) \cdot q^3$']
    input_dists = [input_dist_q0, input_dist_q1, input_dist_q2, input_dist_q3]
    interp_dists = [interp1_dist_q0, interp1_dist_q1, interp1_dist_q2, interp1_dist_q3]

    # Loop through subplots
    for j, ax in enumerate(axes.flat):
        # Plot input data points
        ax.scatter(input_q_arr, input_dists[j][index],
                   color='blue', marker='o', edgecolors='black',
                   alpha=0.8, label='Input Data')

        # Plot first interpolation
        ax.plot(interp1_q_arr, interp_dists[j][index](interp1_q_arr),
                linestyle='--', linewidth=2, color='red',
                label='First Interpolation')

        # Labels and title
        ax.set_xlabel(r'$q$', fontsize=12)
        ax.set_ylabel(titles[j], fontsize=12)
        # ax.set_title(titles[i], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)

        ax.set_xscale('log')
        # ax.set_yscale('log')

        if j == 0:
            ax.set_xlim(left=-0.2, right=5.1)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # --- GIVE SOME BASIC INFO ABOUT STORED DATA ---
    axion_masses = interpolator.get_axion_masses()
    min_axion_mass = min(axion_masses)
    max_axion_mass = max(axion_masses)
    N_distributions = interpolator.get_number_of_distributions()

    print(f"Range of axion masses: [{min_axion_mass},{max_axion_mass}] eV.")
    print(f"Number of different distributions in stored file: {N_distributions}")

    # --- CHECK WHETHER MAXIM DISTRIBUTION HAS NEGATIVE VALUE
    negative_counter = 0

    input_dist_q2 = interpolator.get_input_distribution(dist="f(q)_q2")[index]

    for i in range(0, 200):
        dist_i = input_dist_q2[i]
        if dist_i < 0:
            negative_counter += 1
            print(f"Maxim distribution, index: {index} have negative vale for qi: {i}")

    print("How many negative points:", negative_counter)