"""
This script provides a fitting function approach for calculating relativistic degrees of freedom in the early universe.

We focus on two types of relativistic degrees of freedom:
  - g_eff_e: Energy density degrees of freedom
  - g_eff_s: Entropy density degrees of freedom

This approach is based on the fitting functions developed in paper 1803.01038.

The script allows:
  - Computation of relativistic degrees of freedom using analytical fits.
  - Evaluation of the evolution of g_eff_s and g_eff_e as a function of temperature.
  - Handling of multiple temperature regions with distinct parametrization.

Author: Krzysztof Szafrański
Date: 2025-03-17
"""
import numpy as np

# ####################################### IMPLEMENTATION ############################################################# #
# -------------------------------------------------------------------------------------------------------
# FITTING FUNCTION: RELATIVISTIC DEGREES OF FREEDOM
# -------------------------------------------------------------------------------------------------------
# This approach includes g_star_p, and g_star_s.
# - The values represent different relativistic degrees of freedom as a function of temperature (kB*T).
# - The temperature values (kB_T) are given in electron volts (eV).
# - This approach coincides with paper: 1803.01038

class RelativisticDOFFitModel:
    """
    A class to calculate the relativistic degree of freedom as a function
    of temperature (kB*T) using a fitting function. This approach has been
    developed and presented in: 1803.01038

    Attributes:
        min_kB_T (float): Minimum kB*T value available in the dataset.
        max_kB_T (float): Maximum kB*T value available in the dataset.
    """
    def __init__(self):
        """
        Initializes the class by storing some constants and
        coefficients for the fitting functions.

        Degrees of freedom available:
        - g_eff_n: Effective degrees of freedom for number density.
        - g_eff_e: Effective degrees of freedom for energy density.
        """
        self.dof_types = ["g_eff_s", "g_eff_e"]

        # Fixed particle masses in [GeV]
        self.m_e, self.m_mu, self.m_pi0, self.m_piPlus = 0.511e-3, 0.1056, 0.135, 0.140
        self.m_1, self.m_2, self.m_3, self.m_4 = 0.5, 0.77, 1.2, 2

        # Coefficients for the fitting function
        self.cff_arr = {
            "a_i": np.array([
                1, 1.11724, 0.312672, -0.0468049,
                -0.0265004, -0.0011976, 0.000182812, 0.000136436,
                0.0000855051, 0.000012284, 3.82259E-07, -6.87035E-09
            ]),
            "b_i": np.array([
                0.0143382, 0.0137559, 0.00292108, -0.000538533,
                -0.000162496, -2.87906E-05, -3.84278E-06, 2.78776E-06,
                7.40342E-07, 1.17210E-07, 3.72499E-09, -6.74107E-11
            ]),
            "c_i": np.array([
                1, 0.607869, -0.154485, -0.224034,
                -0.0282147, 0.029062, 0.00686778, -0.00100005,
                -0.000169104, 1.06301E-05, 1.69528E-06, -9.33311E-08
            ]),
            "d_i": np.array([
                70.7388, 91.8011, 33.1892, -1.39779,
                -1.52558, -0.0197857, -0.160146, 8.22615E-05,
                0.0202651, -1.82134E-05, 7.83943E-05, 7.13518E-05
            ])
        }

        # Set the range of kB*T values in [eV]
        self.min_kB_T = 1.0e4  # [eV]
        self.max_kB_T = 1e13   # [eV]

    # ------------------------------------ FITTING FUNCTIONS --------------------------------------------------------- #
    @staticmethod
    def f_p(x):
        return np.exp(-1.04855 * x) * (1 + 1.03757 * x + 0.508630 * x ** 2 + 0.0893988 * x ** 3)

    @staticmethod
    def b_p(x):
        return np.exp(-1.03149 * x) * (1 + 1.03317 * x + 0.398264 * x ** 2 + 0.0648056 * x ** 3)

    @staticmethod
    def f_s(x):
        return np.exp(-1.04190 * x) * (1 + 1.03400 * x + 0.456426 * x ** 2 + 0.0595248 * x ** 3)

    @staticmethod
    def b_s(x):
        return np.exp(-1.03365 * x) * (1 + 1.03397 * x + 0.342548 * x ** 2 + 0.0506182 * x ** 3)

    @staticmethod
    def S_fit(x):
        return 1 + 7 / 4 * np.exp(-1.0419 * x) * (1 + 1.034 * x + 0.456426 * x ** 2 + 0.0595249 * x ** 3)

    # ------------------------------------ COMPUTE DOF --------------------------------------------------------------- #
    def get_degrees_of_freedom(self, kB_T_values, dof_type="g_eff_s"):
        """
        Predicts the relativistic degrees of freedom for a given temperature (kB*T).

        Parameters:
            kB_T_values (float or array-like): Temperature values in units of eV (kB*T).
            dof_type (str): Type of relativistic degree of freedom to compute.
                Options: "g_eff_e", "g_eff_s".
                Default: "g_eff_s" (entropy density degrees of freedom).

        Returns:
            float or np.ndarray: Value(s) of the specified degree of freedom.
        """
        # --- Check type of DOF
        if dof_type not in self.dof_types:
            raise ValueError(f"Invalid degree of freedom type '{dof_type}'. Choose from: {self.dof_types}")

        # --- Ensure kB_T_values is a numpy array for vectorized operations
        kB_T_values = np.atleast_1d(kB_T_values)
        kB_T_GeV = kB_T_values * 1e-9
        results = np.zeros_like(kB_T_GeV)

        # --- First region: kB_T_GeV <= 0.12
        mask_low = kB_T_GeV <= 0.12
        if np.any(mask_low):
            if dof_type == "g_eff_s":
                results[mask_low] = (2.008 +
                                     1.923 * self.S_fit(self.m_e / kB_T_GeV[mask_low]) +
                                     3.442 * self.f_s(self.m_e / kB_T_GeV[mask_low]) +
                                     3.468 * self.f_s(self.m_mu / kB_T_GeV[mask_low]) +
                                     1.034 * self.b_s(self.m_pi0 / kB_T_GeV[mask_low]) +
                                     2.068 * self.b_s(self.m_piPlus / kB_T_GeV[mask_low]) +
                                     4.160 * self.b_s(self.m_1 / kB_T_GeV[mask_low]) +
                                     0.550 * self.b_s(self.m_2 / kB_T_GeV[mask_low]) +
                                     90    * self.b_s(self.m_3 / kB_T_GeV[mask_low]) +
                                     6209  * self.b_s(self.m_4 / kB_T_GeV[mask_low]))
            elif dof_type == "g_eff_e":
                results[mask_low] = (2.030 +
                                     1.353 * (self.S_fit(self.m_e / kB_T_GeV[mask_low]))**(4/3) +
                                     3.495 * self.f_p(self.m_e / kB_T_GeV[mask_low]) +
                                     3.446 * self.f_p(self.m_mu / kB_T_GeV[mask_low]) +
                                     1.050 * self.b_p(self.m_pi0 / kB_T_GeV[mask_low]) +
                                     2.080 * self.b_p(self.m_piPlus / kB_T_GeV[mask_low]) +
                                     4.165 * self.b_p(self.m_1 / kB_T_GeV[mask_low]) +
                                     30.55 * self.b_p(self.m_2 / kB_T_GeV[mask_low]) +
                                     89.40 * self.b_p(self.m_3 / kB_T_GeV[mask_low]) +
                                     8209  * self.b_p(self.m_4 / kB_T_GeV[mask_low]))

        # --- Second region: kB_T_GeV > 0.12
        mask_high = kB_T_GeV > 0.12
        if np.any(mask_high):
            t = np.log(kB_T_GeV[mask_high])
            sum_ai = np.sum(self.cff_arr["a_i"][:, None] * t ** np.arange(len(self.cff_arr["a_i"]))[:, None], axis=0)
            sum_bi = np.sum(self.cff_arr["b_i"][:, None] * t ** np.arange(len(self.cff_arr["b_i"]))[:, None], axis=0)
            sum_ci = np.sum(self.cff_arr["c_i"][:, None] * t ** np.arange(len(self.cff_arr["c_i"]))[:, None], axis=0)
            sum_di = np.sum(self.cff_arr["d_i"][:, None] * t ** np.arange(len(self.cff_arr["d_i"]))[:, None], axis=0)
            g_starE = sum_ai / sum_bi
            g_starS = g_starE / (1 + sum_ci / sum_di)
            results[mask_high] = g_starS if dof_type == "g_eff_s" else g_starE

        return results[0] if results.size == 1 else results

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
            float: Entropy degrees of freedom g_eff_s at the decoupling moment.
        """
        # Compute the temperature at decoupling: kB*T = m_interaction / x_dec
        kB_T_dec = interaction_mass / x_dec

        # Get the interpolated degrees of freedom at decoupling
        g_decoupling_dof = self.get_degrees_of_freedom(kB_T_dec, dof_type=dof_type)

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
