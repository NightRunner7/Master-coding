"""
This script provides a structured dataset and interpolation functions
for relativistic degrees of freedom in the early universe.

We focus on four types of relativistic degrees of freedom:
  - g_eff_n: Number density degrees of freedom
  - g_eff_e: Energy density degrees of freedom
  - g_eff_p: Pressure degrees of freedom
  - g_eff_s: Entropy density degrees of freedom

Data is sourced from Table A1 of 1609.04979, using the "upper values"
for the 150-214 MeV transition region.

The script allows:
  - Interpolation of g_star values using PCHIP for smooth transitions.
  - Computation of degrees of freedom at different temperatures.
  - Calculation of decoupling conditions for axions based on interaction mass.
  - Visualization of the evolution of relativistic degrees of freedom.

Author: Krzysztof Szafrański
Date: 2025-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# -------------------------------------------------------------------------------------------------------
# EXTENDED DATASET: RELATIVISTIC DEGREES OF FREEDOM
# -------------------------------------------------------------------------------------------------------
# This dataset includes g_star_n, g_star_e, g_star_p, and g_star_s.
# - The values represent different relativistic degrees of freedom as a function of temperature (kB*T).
# - The temperature values (kB_T) are given in electron volts (eV).
# - For the 150-214 MeV transition region, we take the "upper" values from Table A1 of 1609.04979.
# - This dataset is specific to one model, while the original paper contains multiple models.

data_extended = {
    "kB_T_eV": [
        1e13, 5e12, 2e12, 1e12, 5e11, 2e11, 1e11, 5e10, 2e10, 1e10, 5e9, 2e9, 1e9,
        5e8, 2.14e8, 2.00e8, 1.90e8, 1.80e8, 1.70e8, 1.60e8, 1.50e8, 1.40e8, 1.30e8,
        1.00e8, 5.0e7, 2.0e7, 1.0e7, 5.0e6, 2.0e6, 1.0e6, 5.0e5, 2.0e5,
        1.0e5, 5.0e4, 2.0e4, 1.0e4
    ],
    "g_star_n": [
        95.50, 95.49, 95.47, 95.39, 95.11, 93.55, 89.89, 83.53, 77.39, 76.20, 75.27, 71.14,
        65.37, 59.69, 55.37, 26.45, 24.14, 22.16, 20.49, 19.09, 17.93, 16.96, 16.16, 14.39,
        11.87, 9.71, 9.50, 9.49, 9.43, 9.22, 8.53, 5.97, 4.03, 3.64, 3.64, 3.64
    ],
    "g_star_e": [
        106.75, 106.75, 106.74, 106.72, 106.61, 105.90, 103.53, 97.40, 88.45, 86.22, 85.60, 82.50,
        76.34, 69.26, 62.49, 50.75, 44.01, 38.27, 33.47, 29.51, 26.31, 23.77, 21.76, 18.00, 14.63,
        11.33, 10.76, 10.74, 10.71, 10.60, 10.16, 7.66, 4.46, 3.39, 3.36, 3.36
    ],
    "g_star_p": [
        106.75, 106.75, 106.73, 106.65, 106.38, 104.75, 100.80, 93.94, 87.22, 85.85, 84.68, 79.69,
        72.97, 66.43, 61.52, 29.62, 27.04, 24.84, 22.98, 21.42, 20.13, 19.05, 18.16, 16.21, 13.40,
        10.99, 10.75, 10.73, 10.65, 10.36, 9.43, 6.20, 3.84, 3.37, 3.36, 3.36
    ],
    "g_star_s": [
        106.75, 106.75, 106.74, 106.70, 106.56, 105.61, 102.85, 96.53, 88.14, 86.13, 85.37, 81.80,
        75.50, 68.55, 62.25, 45.47, 39.77, 34.91, 30.84, 27.49, 24.77, 22.59, 20.86, 17.55, 14.32,
        11.25, 10.76, 10.74, 10.70, 10.56, 10.03, 7.55, 4.78, 3.93, 3.91, 3.91
    ]
}

# ####################################### MAIN CLASS ################################################################# #
class RelativisticDegreesOfFreedom:
    """
    A class to model and interpolate relativistic degrees of freedom
    as functions of temperature (kB*T) using PCHIP interpolation.

    Attributes:
        data (pd.DataFrame): DataFrame storing the tabulated degrees of freedom.
        min_kB_T (float): Minimum kB*T value available in the dataset.
        max_kB_T (float): Maximum kB*T value available in the dataset.
        interp (dict): Dictionary of PCHIP interpolates for each type of relativistic degree of freedom.
    """

    def __init__(self):
        """
        Initializes the class by storing data and setting up interpolation
        functions for different relativistic degrees of freedom.

        Degrees of freedom available:
        - g_eff_n: Effective degrees of freedom for number density.
        - g_eff_e: Effective degrees of freedom for energy density.
        - g_eff_p: Effective degrees of freedom for pressure.
        - g_eff_s: Effective degrees of freedom for entropy density.
        """
        # Store the dataset
        self.data = pd.DataFrame(data_extended)

        # Ensure data is strictly increasing
        self.data_sorted = self.data.sort_values(by="kB_T_eV").drop_duplicates(subset="kB_T_eV")

        # Set the range of kB*T values
        self.min_kB_T = min(self.data_sorted["kB_T_eV"])
        self.max_kB_T = max(self.data_sorted["kB_T_eV"])

        # Interpolation setup: Use PCHIP for smooth interpolation
        self.interp = dict()
        self.interp["g_eff_n"] = PchipInterpolator(self.data_sorted["kB_T_eV"], self.data_sorted["g_star_n"])
        self.interp["g_eff_e"] = PchipInterpolator(self.data_sorted["kB_T_eV"], self.data_sorted["g_star_e"])
        self.interp["g_eff_p"] = PchipInterpolator(self.data_sorted["kB_T_eV"], self.data_sorted["g_star_p"])
        self.interp["g_eff_s"] = PchipInterpolator(self.data_sorted["kB_T_eV"], self.data_sorted["g_star_s"])

    def get_degrees_of_freedom(self, kB_T_values, dof_type="g_eff_s"):
        """
        Predicts the relativistic degrees of freedom for a given temperature (kB*T).

        Parameters:
            kB_T_values (float or array-like): Temperature values in units of eV (kB*T).
            dof_type (str): Type of relativistic degree of freedom to compute.
                            Options: "g_eff_n", "g_eff_e", "g_eff_p", "g_eff_s".
                            Default: "g_eff_s" (entropy density degrees of freedom).

        Returns:
            float or np.ndarray: Interpolated value(s) of the specified degree of freedom.
        """
        if dof_type not in self.interp:
            raise ValueError(f"Invalid degree of freedom type '{dof_type}'. Choose from: {list(self.interp.keys())}")

        return self.interp[dof_type](kB_T_values)

    def compute_decoupling_dof(self, interaction_mass, x_dec):
        """
        Computes the relativistic degrees of freedom at the decoupling moment
        based on the interaction process responsible for axion production.

        Parameters:
            interaction_mass (float): Mass of the interacting particles in eV.
                                      Example: For muon scattering, this should be m_μ.
            x_dec (float): Dimensionless decoupling parameter, defined as x = m/T,
                           where m is the mass of interacting particles.

        Returns:
            float: Interpolated entropy degrees of freedom g_eff_s at the decoupling moment.
        """
        # Compute the temperature at decoupling: kB*T = m_interaction / x_dec
        kB_T_dec = interaction_mass / x_dec

        # Get the interpolated degrees of freedom at decoupling
        g_decoupling_dof = self.interp["g_eff_s"](kB_T_dec)

        return g_decoupling_dof

    @staticmethod
    def compute_decoupling_temperature(interaction_mass, x_dec):
        """
        Computes the decoupling temperature at the decoupling moment
        based on the interaction process responsible for axion production.

        Parameters:
            interaction_mass (float): Mass of the interacting particles in eV.
                                      Example: For muon scattering, this should be m_μ.
            x_dec (float): Dimensionless decoupling parameter, defined as x = m/T,
                           where m is the mass of interacting particles.

        Returns:
            float: Interpolated entropy degrees of freedom g_eff_s at the decoupling moment.
        """
        # Compute the temperature at decoupling: kB*T = m_interaction / x_dec
        kB_T_dec = interaction_mass / x_dec  # [eV]

        return kB_T_dec


    # ----------------------------------------- ACCESSORS ---------------------------------------------------------- #
    def get_min_kB_T(self):
        """
        Returns the minimum temperature (kB*T) available in the dataset.

        Returns:
            float: Minimum kB*T value in eV.
        """
        return self.min_kB_T

    def get_max_kB_T(self):
        """
        Returns the maximum temperature (kB*T) available in the dataset.

        Returns:
            float: Maximum kB*T value in eV.
        """
        return self.max_kB_T

# ####################################### CROSS CHECKS ############################################################### #
if __name__ == "__main__":
    """
    Generates and displays a plot of relativistic degrees of freedom as a function of kB*T.
    The x-axis and y-axis are properly formatted with log-scaled scientific notation ticks.
    """
    # Initialize the class
    RelativisticDOF = RelativisticDegreesOfFreedom()

    # ------------------------------ CHECK: AXION MASS IN MUON SCATTERING -------------------------------------------- #
    # Typical range of axion mass
    muon_mass = 105.66*10**6  # [eV]
    muon_mass_MeV = 105.66  # [MeV]
    x_dec_muon = 30  # decouple from the plasma
    kB_T_today = 8.617 * 10**(-5) * 2.725  # kB * T, [kB*T] = [kB] * [T] = [eV/K] * [K] = [eV]

    # calculate decoupling dof
    axion_decouple_dof = RelativisticDOF.compute_decoupling_dof(muon_mass, x_dec_muon)
    g_s_today = RelativisticDOF.get_degrees_of_freedom(kB_T_today, dof_type="g_eff_s")
    T_dec_muon = RelativisticDOF.compute_decoupling_temperature(muon_mass, x_dec_muon)
    print(f"muon mass: {muon_mass_MeV} [MeV], corresponds to: {axion_decouple_dof} relativistic dof of entropy")
    print(f"Today corresponds to: {g_s_today} relativistic dof of entropy")
    print(f"Decouple temperatures corresponds to: {T_dec_muon*10**(-6)} MeV")

    # ------------------------------ PLOT FROM: 1609.04979 ----------------------------------------------------------- #
    # Get the range of kB*T values
    min_kBT = RelativisticDOF.get_min_kB_T()
    max_kBT = RelativisticDOF.get_max_kB_T()

    # Generate kB_T values for plotting in MeV
    kB_T_values = np.logspace(np.log10(min_kBT), np.log10(max_kBT), 200)
    kB_T_values_MeV = kB_T_values / 1e6  # Convert eV to MeV

    # Compute interpolated values
    g_n_values = RelativisticDOF.get_degrees_of_freedom(kB_T_values, dof_type="g_eff_n")
    g_e_values = RelativisticDOF.get_degrees_of_freedom(kB_T_values, dof_type="g_eff_e")
    g_p_values = RelativisticDOF.get_degrees_of_freedom(kB_T_values, dof_type="g_eff_p")
    g_s_values = RelativisticDOF.get_degrees_of_freedom(kB_T_values, dof_type="g_eff_s")

    # Create the plot with improved formatting
    plt.figure(figsize=(10, 6))

    # Plot all degrees of freedom
    plt.plot(kB_T_values_MeV[::-1], g_n_values[::-1], label=r"$g_{\star n}$", linestyle=":", linewidth=2,
             color="purple")
    plt.plot(kB_T_values_MeV[::-1], g_e_values[::-1], label=r"$g_{\star e}$", linestyle="-", linewidth=2, color="red")
    plt.plot(kB_T_values_MeV[::-1], g_p_values[::-1], label=r"$g_{\star p}$", linestyle="--", linewidth=2, color="blue")
    plt.plot(kB_T_values_MeV[::-1], g_s_values[::-1], label=r"$g_{\star s}$", linestyle="-.", linewidth=2,
             color="green")

    # Log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Axis labels and title
    plt.xlabel(r"$k_B T$ [MeV]", fontsize=18)
    plt.ylabel(r"$g_{\star}$", fontsize=18)
    plt.title(r"Evolution of Relativistic Degrees of Freedom", fontsize=18)

    # Invert x-axis so that higher temperatures are on the left, lower on the right
    plt.gca().invert_xaxis()

    # Set properly formatted log ticks for x-axis (kB*T in MeV)
    x_ticks = [10 ** i for i in range(6, -3, -1)]  # 10^6 to 10^-2
    plt.xticks(x_ticks, [rf"$10^{{{int(np.log10(tick))}}}$" for tick in x_ticks], fontsize=14)

    # Set properly formatted log ticks for y-axis (Degrees of Freedom)
    y_ticks = [1, 10, 100]
    plt.yticks(y_ticks, [rf"$10^{{{int(np.log10(tick))}}}$" for tick in y_ticks], fontsize=14)

    # Add grid lines
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Show legend
    plt.legend(fontsize=14)

    # Display the plot
    plt.show()