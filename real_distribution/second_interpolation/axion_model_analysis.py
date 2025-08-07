import numpy as np
from scipy.integrate import quad
# --- FROM EXTERNAL FILES ---
from second_interpolation import get_process
from relativistic_dof import RelativisticDOFRegistry
from distribution_first_interpolation import FirstInterpolation

class AxionModelDistribution(FirstInterpolation):
    # ----------------------------------------- INITIALIZATION ------------------------------------------------------ #
    def __init__(self, process_name, file_path, particle_mass):
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

        # --- Load production process dynamically (module with stateful API), Store available fitting functions ---
        self.process = get_process(process_name)
        self.fitting_functions = self._wrap_process_defaults()

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
        self.con["x_dec"] = 30       # axion decoupling scale

        # --- Other Constants ---
        self.con["particle_mass"] = particle_mass  # [eV]

        # ### Compute Relativistic Degrees of Freedom Ratios ######################################################### #
        self.con["g_star_s_today"] = self.RelativisticDOF.get_degrees_of_freedom(self.con["T_photon"]["[eV]"],
                                                                                 dof_type="g_eff_s")
        self.con["g_star_s_axion_decoupling"] = self.RelativisticDOF.compute_decoupling_dof(self.con["particle_mass"],
                                                                                            self.con["x_dec"])

        # ### Physical Constants ##################################################################################### #
        self.con["rho_crit"] = 1.053672 * 10**4    # critical density, [h^2 eV * cm^-3]
        self.con["s0"] = 2 * np.pi**2 / 45 * self.con["g_star_s_today"] * self.con["T_photon"]["[cm^-1]"]**3  # entropy density, [cm^-3]

        # --- Limits ---
        self.con["q_max"] = 19.99
        self.con["q_min"] = 0.01

        # --- Find fa and ΔN_eff ---
        self.output = self.calculate_delta_n_eff()

    # ----------------------------------------- BASE ----------------------------------------------------------------- #
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
        return q ** 2 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** (-1)

    @staticmethod
    def f_approx_q3(q, A, b, mu):
        """
        Approximate function for fitting f(q) * q^2.

        Returns:
            Evaluated function values at q.
        """
        return q ** 3 * (np.exp(A * np.sqrt(1 + q ** 2) - b) + mu) ** (-1)

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
        A = self.fitting_functions["A"](log_axion_mass)
        b = self.fitting_functions["b"](log_axion_mass)
        mu = self.fitting_functions["mu"](log_axion_mass)

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
    def calculate_delta_n_eff(self):
        """
        Compute the axion contribution to the extra relativistic degrees of freedom (ΔN_eff)
        and comoving number density (Ya) for each axion decay constant (f_a).

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
        q_min = self.con["q_min"]

        # --- Select integrand functions
        integrand_q3 = self.f_approx_q3
        integrand_q2 = self.f_approx_q2

        # --- Initialize result storage
        N = self.distributions["input"]["N"]
        fa_arr = self.param["input"]["fa_arr"]
        log_ma_arr = np.log(self.param["input"]["ma_arr"])
        delta_neff_arr = np.zeros(N)
        ya_arr = np.zeros(N)

        # --- Perform integration for each f_a
        for i in range(N):
            log_ma = log_ma_arr[i]
            val_A = self.fitting_functions["A"](log_ma)
            val_b = self.fitting_functions["b"](log_ma)
            val_mu = self.fitting_functions["mu"](log_ma)
            integral_q3, _ = quad(integrand_q3, q_min, q_max, args=(val_A, val_b, val_mu), epsabs=1e-10, epsrel=1e-10, limit=500)
            integral_q2, _ = quad(integrand_q2, q_min, q_max, args=(val_A, val_b, val_mu), epsabs=1e-10, epsrel=1e-10, limit=500)
            delta_neff_arr[i] = const_delta_neff * integral_q3
            ya_arr[i] = const_ya * integral_q2
            # ya_arr[i] = (delta_neff_arr[i] / 75.64)**(3/4)

        return {
            "fa": fa_arr,
            "delta_neff": delta_neff_arr,
            "ya": ya_arr
        }

    # ----------------------------------------- GET DATA ------------------------------------------------ #
    def get_fitted_parameters(self, m_a):
        """
        Retrieve the fitted parameters A, b, and μ for a given axion mass.

        Parameters:
        -----------
        - m_a (float): Axion mass in eV.

        Returns:
        --------
        tuple: (A, b, μ) best-fit parameters for the given axion mass.
        """
        log_m_a = np.log(m_a)
        return (
            self.fitting_functions["A"](log_m_a),
            self.fitting_functions["b"](log_m_a),
            self.fitting_functions["mu"](log_m_a)
        )

    def get_physical_constant(self, physical_constant_name):
        """Return one of the values from the physical constant dictionary"""
        if physical_constant_name not in self.con.keys():
            raise ValueError(f"Invalid physcial constant name '{physical_constant_name}'. Choose from: {list(self.con.keys())}")

        return self.con[physical_constant_name]

    def get_neff_and_ya_vs_fa(self):
        """Return fa and calculated extra relativistic degrees of freedom for this fa"""
        return self.output

    def get_ma_fixed_wa(self, fixed_wa):
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
