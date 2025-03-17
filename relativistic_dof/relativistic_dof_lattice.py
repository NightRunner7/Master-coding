"""
This script provides a fitting function approach for calculating relativistic degrees of freedom in the early universe.

We focus on two types of relativistic degrees of freedom:
  - g_eff_e: Energy density degrees of freedom
  - g_eff_s: Entropy density degrees of freedom

This approach is based on the fitting functions developed in paper 1606.07494 (Lattice QFT).

The script allows:
  - Computation of relativistic degrees of freedom using analytical fits and interpolation.
  - Evaluation of the evolution of g_eff_s and g_eff_e as a function of temperature.
  - Handling of multiple temperature regions with distinct parametrization.

Author: Krzysztof Szafrański
Date: 2025-03-17
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# ####################################### IMPLEMENTATION ############################################################# #
# -------------------------------------------------------------------------------------------------------
# EXTENDED DATASET: RELATIVISTIC DEGREES OF FREEDOM
# -------------------------------------------------------------------------------------------------------
# This approach includes g_star_p, and g_star_s.
# - The values represent different relativistic degrees of freedom as a function of temperature (kB*T).
# - The temperature values (kB_T) are given in electron volts (eV).
# - This approach coincides with paper: 1606.07494

class RelativisticDOFLattice:
    """
    A class to model and interpolate relativistic degrees of freedom
    as functions of temperature (T) using PCHIP interpolation based on Table S2 from 1606.07494.
    """
    def __init__(self):
        """
        Initializes the class by storing data and setting up interpolation functions
        for different relativistic degrees of freedom.
        """
        # --- Table S2 from 1606.07494
        data_spline = {
            "log10_kB_T_MeV": [
                0.00, 0.50, 1.00, 1.25, 1.60, 2.00, 2.15, 2.20,
                2.40, 2.50, 3.00, 4.00, 4.30, 4.60, 5.00, 5.45
            ],
            "g_rho": [
                10.71, 10.74, 10.76, 11.09, 13.68, 17.61, 24.07, 29.84,
                47.83, 53.04, 73.48, 83.10, 85.56, 91.97, 102.17, 104.98
            ],
            "g_rho_g_s_ratio": [
                1.00228, 1.00029, 1.00048, 1.00505, 1.02159, 1.02324, 1.05423, 1.07578,
                1.06118, 1.04690, 1.01778, 1.00123, 1.00389, 1.00887, 1.00750, 1.00023
            ]
        }

        # --- Convert log scale temperature back to linear scale in MeV
        self.data = pd.DataFrame(data_spline)
        self.data["kB_T_MeV"] = 10 ** self.data["log10_kB_T_MeV"]

        # --- Compute g_s from the ratio: g_s = g_rho / (g_rho / g_s)
        self.data["g_s"] = self.data["g_rho"] / self.data["g_rho_g_s_ratio"]

        # --- Setup PCHIP interpolation
        self.interp = {
            "g_eff_e": PchipInterpolator(self.data["kB_T_MeV"], self.data["g_rho"]),
            "g_eff_s": PchipInterpolator(self.data["kB_T_MeV"], self.data["g_s"])
        }

        # --- Store temperature range
        self.min_kB_T = min(self.data["kB_T_MeV"]) * 1e6  # [eV]
        self.max_kB_T = max(self.data["kB_T_MeV"]) * 1e6  # [eV]

    def get_degrees_of_freedom(self, kB_T_values, dof_type="g_eff_s"):
        """
        Predicts the relativistic degrees of freedom for a given temperature (T in MeV).

        Parameters:
            kB_T_values (float or array-like): Temperature values in units of eV (kB*T).
            dof_type (str): Type of relativistic degree of freedom to compute.
                            Options: "g_eff_e" (energy), "g_eff_s" (entropy).

        Returns:
            float or np.ndarray: Interpolated value(s) of the specified degree of freedom.
        """
        if dof_type not in self.interp:
            raise ValueError(f"Invalid dof_type '{dof_type}'. Choose from {list(self.interp.keys())}")

        # Convert input to NumPy array for interpolation
        kB_T_MeV = np.atleast_1d(kB_T_values * 1e-6)

        return self.interp[dof_type](kB_T_MeV)

    def compute_decoupling_dof(self, interaction_mass, x_dec, dof_type="g_eff_s"):
        """
        Computes the relativistic degrees of freedom at the decoupling moment
        based on the interaction process responsible for axion production.

        Parameters:
            interaction_mass (float): Mass of the interacting particles in eV.
                                      Example: For muon scattering, this should be m_μ.
            x_dec (float): Dimensionless decoupling parameter, defined as x = m/T,
                           where m is the mass of interacting particles.
            dof_type (str): Type of relativistic degree of freedom to compute.
                Options: "g_eff_e", "g_eff_s".
                Default: "g_eff_s" (entropy density degrees of freedom).

        Returns:
            float: Interpolated entropy degrees of freedom g_eff_s at the decoupling moment.
        """
        # Compute the temperature at decoupling: kB*T = m_interaction / x_dec
        kB_T_dec = interaction_mass / x_dec

        # Convert into proper units [MeV]
        kB_T_dec_MeV = kB_T_dec * 1e-6

        # Get the interpolated degrees of freedom at decoupling
        g_decoupling_dof = self.interp[dof_type](kB_T_dec_MeV)

        return np.float64(g_decoupling_dof)

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
            float: The decoupling temperature at the decoupling moment.
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