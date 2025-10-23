import numpy as np
from scipy.integrate import quad
# --- FROM EXTERNAL FILES ---
from relativistic_dof import RelativisticDOFRegistry
from distribution_first_interpolation import FirstInterpolation

class AxionModelMaximDistribution(FirstInterpolation):
    """
    Class for handling axion distribution function coming from numerical solving Boltzmann equation,
    which is job done by Maxim Laletin.

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
    def __init__(self, file_path, particle_mass, x_dec):
        """
        Initialize axion model class for taon decay.

        Parameters:
        -----------
        file_path (str): Path to the data file containing axion distributions.
        particle_mass (float): Mass in [eV] of particles involve in axion production.
        x_dec (float): Axion decoupling scale, typically in range [20, 30].
        """
        # --- Inherit from FirstInterpolation (load data and perform first interpolate of distributions) ---
        super().__init__(file_path)

        # --- Relativistic Degrees of Freedom Handler ---
        self.RelativisticDOF = RelativisticDOFRegistry.get_method("fit")

        # ### Physical Constants ##################################################################################### #
        self.con = dict()
        self.con["hbar"] = 6.582119569 * 10**(-16) # hbar, [eV * s]
        self.con["kB"] = 8.617333262 * 10**(-5)    # Boltzmann constant, [eV / K]
        self.con["c"] = 2.99792458 * 10**8         # speed of light, [m / s]
        convert_ev_to_cm = 1/(self.con["hbar"] * self.con["c"] * 10**2)  # [cm^-1]

        # --- Constants corresponding to photon ---
        self.con["g_dof_photon"] = 2  # photon degrees of freedom
        # Photon temperature today (nowadays)
        self.con["T_photon"] = dict()
        self.con["T_photon"]["[K]"] = 2.7255                      # [K] in Kelvin
        self.con["T_photon"]["[eV]"] = 2.7255 * self.con["kB"]    # kB * T, [kB*T] = [kB] * [T] = [eV/K] * [K] = [eV]
        self.con["T_photon"]["[cm^-1]"] = 2.7255 * self.con["kB"] * convert_ev_to_cm # [cm^-1]
        self.con["T_photon"]["[T_photon]"] = 1                    # [T_photon]

        # --- Constants corresponding to axion ---
        self.con["g_dof_axion"] = 1  # axion degrees of freedom
        self.con["x_dec"] = x_dec    # axion decoupling scale

        # --- Other Constants ---
        self.con["particle_mass"] = particle_mass  # [eV]

        # ### Compute Relativistic Degrees of Freedom Ratios ######################################################### #
        self.con["g_star_s_today"] = self.RelativisticDOF.get_degrees_of_freedom(self.con["T_photon"]["[eV]"],
                                                                                 dof_type="g_eff_s")
        self.con["g_star_s_axion_decoupling"] = self.RelativisticDOF.compute_decoupling_dof(self.con["particle_mass"],
                                                                                            self.con["x_dec"])
        # FOR TAON DECAY: MAXIM SETTING
        # self.con["g_star_s_today"] = 43/11
        # self.con["g_star_s_axion_decoupling"] = 15.4185

        # ### Physical Constants ##################################################################################### #
        self.con["rho_crit"] = 1.053672 * 10**4    # critical density, [h^2 eV * cm^-3]
        self.con["s0"] = 2 * np.pi**2 / 45 * self.con["g_star_s_today"] * self.con["T_photon"]["[cm^-1]"]**3  # entropy density, [cm^-3]

        # --- Limits ---
        self.con["q_max"] = 19.99

        # --- Find fa and ΔN_eff ---
        self.output = self.calculate_delta_n_eff(use_rescaled_q=False)

    # ----------------------------------------- BASE FUNCTIONS ------------------------------------------------------- #
    def distribution_fq_q3_tilda(self, q_tilda, index):
        """
        Compute the axion distribution function f(q̃) * q̃³ with rescaled momentum q̃.

        Parameters:
            q_tilda (float): Rescaled co-moving momentum (q̃).
            index (int): Index to select the interpolation corresponding to a specific axion mass.

        Returns:
            float: Approximated value of f(q̃) * q̃³.
        """
        q_val = self._rescale_q(q_tilda)
        return self.distributions["interp1"]["f(q)_q2"][index](q_val) * q_val

    def distribution_fq_q2_tilda(self, q_tilda, index):
        """
        Compute the axion distribution function f(q̃) * q̃² with rescaled momentum q̃.

        Parameters:
            q_tilda (float): Rescaled co-moving momentum (q̃).
            index (int): Index to select the interpolation corresponding to a specific axion mass.

        Returns:
            float: Approximated value of f(q̃) * q̃².
        """
        q_val = self._rescale_q(q_tilda)
        return self.distributions["interp1"]["f(q)_q2"][index](q_val)

    def distribution_fq_q3(self, q_val, index):
        """
        Compute the axion distribution function f(q) * q³ (without rescaling).

        Parameters:
            q_val (float): Co-moving momentum q.
            index (int): Index to select the interpolation corresponding to a specific axion mass.

        Returns:
            float: Approximated value of f(q) * q³.
        """
        return self.distributions["interp1"]["f(q)_q2"][index](q_val) * q_val

    def distribution_fq_q2(self, q_val, index):
        """
        Compute the axion distribution function f(q) * q² (without rescaling).

        Parameters:
            q_val (float): Co-moving momentum q.
            index (int): Index to select the interpolation corresponding to a specific axion mass.

        Returns:
            float: Approximated value of f(q) * q².
        """
        return self.distributions["interp1"]["f(q)_q2"][index](q_val)

    def _rescale_q(self, q_tilda):
        """
        Rescales q̃ to q using entropy degrees of freedom.

        Parameters:
            q_tilda (float): Rescaled momentum q̃.

        Returns:
            float: Original momentum q.
        """
        g_star_s_ratio = self.con["g_star_s_axion_decoupling"] / self.con["g_star_s_today"]
        return q_tilda * g_star_s_ratio ** (-1 / 3)

    # ----------------------------------------- CALCULATE DELTA N_eff ------------------------------------------------ #
    def calculate_delta_n_eff(self, use_rescaled_q=False):
        """
        Compute the axion contribution to the extra relativistic degrees of freedom (ΔN_eff)
        and comoving number density (Ya) for each axion decay constant (f_a).

        Parameters:
            use_rescaled_q (bool): Whether to compute using rescaled momenta (q̃)
                                   instead of physical q. Defaults to False.

        Returns:
            dict: Dictionary with arrays for each f_a value containing:
                  - 'fa': Axion decay constants.
                  - 'delta_neff': Corresponding ΔN_eff values.
                  - 'ya': Corresponding comoving axion number densities.
        """
        # --- Precompute ratios and constants
        g_star_s_ratio = self.con["g_star_s_axion_decoupling"] / self.con["g_star_s_today"]
        g_dof_ratio = self.con["g_dof_axion"] / self.con["g_dof_photon"]

        const_delta_neff = (
                8 / 7 * (11 / 4) ** (4 / 3) * 15 / (np.pi ** 4) * g_dof_ratio * g_star_s_ratio ** (-4 / 3)
        )
        const_ya = (
                45 / (4 * np.pi ** 4) * g_star_s_ratio ** (-1) / self.con["g_star_s_today"]
        )

        # --- Integration limits
        q_max = self.con["q_max"]
        q_tilda_upper_limit = q_max * g_star_s_ratio ** (1 / 3)

        # --- Select integrand functions
        if use_rescaled_q:
            integrand_q3 = self.distribution_fq_q3_tilda
            integrand_q2 = self.distribution_fq_q2_tilda
            q_limit = q_tilda_upper_limit
        else:
            integrand_q3 = self.distribution_fq_q3
            integrand_q2 = self.distribution_fq_q2
            q_limit = q_max

        # --- Initialize result storage
        N = self.distributions["input"]["N"]
        fa_arr = np.array(self.param["input"]["fa_arr"])
        delta_neff_arr = np.zeros(N)
        ya_arr = np.zeros(N)

        # --- Perform integration for each f_a
        for i in range(N):
            integral_q3, _ = quad(integrand_q3, 0, q_limit, args=i, limit=200)
            integral_q2, _ = quad(integrand_q2, 0, q_limit, args=i, limit=200)

            delta_neff_arr[i] = const_delta_neff * integral_q3
            ya_arr[i] = const_ya * integral_q2
            # ya_arr[i] = (delta_neff_arr[i] / 75.64)**(3/4)

        return {
            "fa": fa_arr,
            "delta_neff": delta_neff_arr,
            "ya": ya_arr
        }

    # ----------------------------------------- GET DATA ------------------------------------------------ #
    def get_physical_constant(self, physical_constant_name):
        """Return one of the values from the physical constant dictionary"""
        if physical_constant_name not in self.con.keys():
            raise ValueError(f"Invalid physcial constant name '{physical_constant_name}'. Choose from: {list(self.con.keys())}")

        return self.con[physical_constant_name]

    def get_neff_and_ya_vs_fa(self):
        """Return fa and calculated extra relativistic degrees of freedom for this fa"""
        return self.output

    def get_ma_fixed_wa(self, fixed_wa, fitMethod=False):
        """
        Return the axion mass array m_a[i] (eV) required to produce a given
        physical density parameter fixed_wa = Ω_a h^2, for each entry in
        self.output["ya"] = Y_a[i] = n_a/s (dimensionless yield).

        Physics:
            ω_a = Ω_a h^2 = (m_a * Y_a * s0) / (ρ_crit * h^2)
            -> m_a = ω_a * ρ_crit / (Y_a * s0)

        Assumes:
            self.con["s0"]        : present-day entropy density (number/volume), [cm^-3]
            self.con["rho_crit"]  : critical density *in the same convention*
                                    used in the ω_a definition (see notes above), [h^2 * eV * cm^-3]
        """
        if fitMethod:
            Y = np.asarray(self.output["ya"], dtype=float)
            factor = self.con["rho_crit"] / self.con["s0"]  # [eV]
            # Fit coef
            A = 0.14
            b = -1.1
            omega_m = 0.3157 * 0.6745**2
            # Vectorized computation
            ma_arr = (A * 1000**b * omega_m * factor/Y)**(1/(1+b)) # [eV]
            return ma_arr
        else:
            Y = np.asarray(self.output["ya"], dtype=float)

            if np.any(Y <= 0):
                raise ValueError("Encountered non-positive yield(s) in self.output['ya']; cannot compute mass.")

            factor = self.con["rho_crit"] / self.con["s0"]  # [eV]

            # Vectorized computation
            ma_arr = fixed_wa * factor / Y  # [eV]
            return ma_arr

    # ----------------------------------------- CHANGE SETTINGS ----------------------------------------- #
    def change_physical_constant(self, physical_constant_name, physical_constant_value):
        """Function to change one physical constant / setting of simulation"""
        if physical_constant_name=="g_star_s_today":
            self.con[physical_constant_name] = physical_constant_value
        elif physical_constant_name=="g_star_s_axion_decoupling":
            self.con[physical_constant_name] = physical_constant_value
        elif physical_constant_name=="g_dof_axion":
            self.con[physical_constant_name] = physical_constant_value
        elif physical_constant_name=="g_dof_photon":
            self.con[physical_constant_name] = physical_constant_value
        elif physical_constant_name=="x_dec":
            self.con[physical_constant_name] = physical_constant_value
            self.con["g_star_s_axion_decoupling"] = self.RelativisticDOF.compute_decoupling_dof(self.con["particle_mass"],
                                                                                                self.con["x_dec"])
