import numpy as np
from scipy.integrate import quad
# --- FROM EXTERNAL FILES ---
from relativistic_dof import RelativisticDOFRegistry
from axion_production.distribution_first_interpolation import FirstInterpolation

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
        self.RelativisticDOF = RelativisticDOFRegistry.get_method("lattice")

        # --- Physical Constants ---
        self.con = dict()  # dictionary with constants
        self.con["particle_mass"] = particle_mass  # [eV]
        self.con["kB_T_today"] = 8.617 * 10**(-5) * 2.725  # kB * T, [kB*T] = [kB] * [T] = [eV/K] * [K] = [eV]
        self.con["x_dec"] = x_dec  # Axion decoupling scale
        self.con["g_dof_axion"] = 1  # axion degrees of freedom
        self.con["g_dof_photon"] = 2  # photon degrees of freedom

        # --- Compute Relativistic Degrees of Freedom Ratios ---
        self.con["g_star_s_today"] = self.RelativisticDOF.get_degrees_of_freedom(self.con["kB_T_today"],
                                                                                 dof_type="g_eff_s")
        self.con["g_star_s_axion_decoupling"] = self.RelativisticDOF.compute_decoupling_dof(self.con["particle_mass"],
                                                                                            self.con["x_dec"])
        # FOR TAON DECAY: MAXIM SETTING
        # self.con["g_star_s_today"] = 43/11
        # self.con["g_star_s_axion_decoupling"] = 15.4185

        # --- Limits ---
        self.con["q_max"] = 19.99

        # --- Find fa and ΔN_eff ---
        self.output = self.calculate_delta_n_eff()

    # ----------------------------------------- CALCULATE DELTA N_eff ------------------------------------------------ #
    def calculate_delta_n_eff(self, tilda_dist=False):
        """
        Compute the extra relativistic degrees of freedom (ΔN_eff) for axion mass m_a.

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
        # Define upper integration limit for q_tilda
        q_tilda_upper_limit = self.con["q_max"] * g_star_s_ratio ** (1 / 3)

        # Define the distribution function with the scaled momentum q_tilda
        def distribution_tilda(q_tilda, index):
            """
            Computes the approximated axion distribution function in terms of q_tilda.
            """
            # Rescale q_tilda to get the original q value
            q_val = q_tilda * g_star_s_ratio ** (-1 / 3)
            # Compute the approximate distribution value using fitted parameters
            return self.distributions["interp1"]["f(q)_q2"][index](q_val) * q_val

        # Define the distribution function
        def distribution(q_val, index):
            """
            Computes the approximated axion distribution function in terms of q_tilda.
            """
            # Compute the approximate distribution value using fitted parameters
            return self.distributions["interp1"]["f(q)_q2"][index](q_val) * q_val

        # --- Perform numerical integration for each values of decay constant: fa
        discrete_data = dict()
        discrete_data["fa"] = np.array([])
        discrete_data["delta_neff"] = np.array([])

        for i in range(0, self.distributions["input"]["N"]):
            # Select way of integrating
            if tilda_dist:
                integral_of_dist, _ = quad(distribution_tilda, 0, q_tilda_upper_limit, args=i)
            else:
                integral_of_dist, _ = quad(distribution, 0, self.con["q_max"], args=i)
            # find fa and ΔN_eff
            fa_val = self.param["input"]["fa_arr"][i]
            delta_neff_val = const * integral_of_dist
            # appending
            discrete_data["fa"] = np.append(discrete_data["fa"], fa_val)
            discrete_data["delta_neff"] = np.append(discrete_data["delta_neff"], delta_neff_val)

        return discrete_data

    # ----------------------------------------- GET DATA ------------------------------------------------ #
    def get_physical_constant(self, physical_constant_name):
        """Return one of the values from the physical constant dictionary"""
        if physical_constant_name not in self.con.keys():
            raise ValueError(f"Invalid physcial constant name '{physical_constant_name}'. Choose from: {list(self.con.keys())}")

        return self.con[physical_constant_name]

    def get_fa_and_delta_neff(self):
        """Return fa and calculated extra relativistic degrees of freedom for this fa"""
        return self.output

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

# ####################################### CROSS CHECKS ############################################################### #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # --- SELECT FILES ---
    file_number = 3

    # --- ALL STORED FILES ---
    title_arr = [r"$\bf e$ scattering", r"$\bf \mu$ decay", r"$\bf \mu$ scattering",
                 r"$\bf \tau$ decay", r"$\bf \tau$ scattering"]
    filename_neff_arr = ["dNeff_fa_e_scat.dat", "dNeff_fa_mu_dec.dat", "dNeff_fa_mu_scat.dat",
                         "dNeff_fa_tau_dec.dat", "dNeff_fa_tau_scat.dat"]
    filename_dist_arr = ["Distributions_fa_e_scat.dat", "Distributions_fa_mu_dec.dat", "Distributions_fa_mu_scat.dat",
                         "Distributions_fa_tau_dec.dat", "Distributions_fa_tau_scat.dat"]
    filename_neff = f"../{filename_neff_arr[file_number]}"  # File with fa and ΔN_eff
    filename_dist = f"../{filename_dist_arr[file_number]}"  # File with distribution

    # --- FIXED PARTICLES MASSES AND DECOUPLE MOMENT ---
    taon_mass = 1777 * 10**6  # [eV]
    muon_mass = 105.66 * 10**6  # [eV]
    electron_mass = 511 * 10**3  # [eV]
    particle_arr = [electron_mass, muon_mass, muon_mass, taon_mass, taon_mass]
    x_decouple = 30

    # --- CREATE CLASS TO INTEGRATE DISTRIBUTION ---
    axionModel = AxionModelMaximDistribution(filename_dist, particle_arr[file_number], x_decouple)
    # get the data of our interest
    our_data = axionModel.get_fa_and_delta_neff()

    # --- IMPORTING MAXIM DELTA NEFF CALCULATIONS ---
    # Read data while skipping the first row (header)
    data = np.loadtxt(filename_neff, delimiter=",", skiprows=1)

    # --- PRINTING SOME INFORMATION ---
    # Extract the first and third columns
    fa_arr_MAXIM = data[:, 0]  # First column
    delta_Neff_MAXIM = data[:, 2]  # Third column

    N_distribution_Maxim = len(fa_arr_MAXIM)
    print("N_distribution_Maxim:", N_distribution_Maxim)
    decouple_dof_g_s = axionModel.get_physical_constant("g_star_s_axion_decoupling")
    print("g_star_s_axion_decoupling:", decouple_dof_g_s)

    # ---------------------------- DO COMPARISON PLOT: MAXIM RESULTS VS OUR INTEGRATION ------------------------------ #
    fig, axs = plt.subplots(figsize=(8, 6))

    # Plot: f_a vs Delta_N_eff with larger font size and thicker lines
    axs.plot(fa_arr_MAXIM, delta_Neff_MAXIM, color='b', lw=3,
             label=r'$\Delta N_{\rm{eff}}$: MAXIM')
    axs.plot(our_data["fa"], our_data["delta_neff"], color='r', lw=3, linestyle="--",
             label=r'$\Delta N_{\rm{eff}}$: OUR')
    axs.set_xscale('log')
    axs.set_xlabel(r'Decay Constant $\bf f_a$ [GeV]', fontsize=14, fontweight='bold')
    axs.set_ylabel(r'$\bf \Delta N_{eff}$', fontsize=14, fontweight='bold')
    axs.set_title(title_arr[file_number], fontsize=16, fontweight='bold')
    axs.grid(True, linestyle='--', linewidth=0.5)
    axs.legend(fontsize=12)

    # --- Dashed-line
    # Dashed-dotted line value for Planck 2018 constraint
    planck_limit = 0.33

    # Add dashed-dotted line for Planck 2018 limit
    axs.axhline(planck_limit, color='gray', linestyle='-.', lw=2)

    # Add text for Planck 2018 directly on the plot
    axs.text(fa_arr_MAXIM[-1], planck_limit - 0.025, 'Planck 2018', color='gray', fontsize=12, va='bottom', ha='right')

    # Fill the entire region above the Planck limit with gray shading
    ymin, ymax = axs.get_ylim()
    xmin, xmax = axs.get_xlim()
    axs.set_xlim(xmin, xmax)
    axs.set_ylim(-0.05, ymax)
    xaxes = np.linspace(xmin, xmax, num=50)
    axs.fill_between(xaxes, planck_limit, ymax, color='gray', alpha=0.3)

    # --- Other settings
    # Set x-axis ticks in log scale manner with explicit scientific formatting and increase tick size
    axs.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    axs.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'$10^{{{int(np.log10(val))}}}$'))
    axs.tick_params(axis='x', which='major', labelsize=12)  # Increase x-axis tick size

    # --- SAVING PLOT ---
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('Maxin_Vs_Our_delta_Neff.png', format='png', dpi=300)
    # plt.close()

    # ---------------------------- DO COMPARISON PLOT (LOG): MAXIM RESULTS VS OUR INTEGRATION ------------------------ #
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot: f_a vs Delta_N_eff with larger font size and thicker lines
    ax.plot(fa_arr_MAXIM, delta_Neff_MAXIM, color='b', lw=3,
             label=r'$\Delta N_{\rm{eff}}$: MAXIM')
    ax.plot(our_data["fa"], our_data["delta_neff"], color='r', lw=3, linestyle="--",
             label=r'$\Delta N_{\rm{eff}}$: OUR')

    # --- Other settings
    # Set xy-axis to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # --- Labels and title with bold text
    ax.set_xlabel(r'Decay Constant $\bf f_a$ [GeV]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\bf \Delta N_{eff}$', fontsize=14, fontweight='bold')
    ax.set_title(title_arr[file_number], fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)

    # --- Dashed-line
    # Dashed-dotted line value for Planck 2018 constraint
    planck_limit = 0.33

    # Add dashed-dotted line for Planck 2018 limit
    ax.axhline(planck_limit, color='gray', linestyle='-.', lw=2)

    # Add text for Planck 2018 directly on the plot
    ax.text(fa_arr_MAXIM[-1], planck_limit * 0.8, 'Planck 2018', color='gray', fontsize=12, va='bottom', ha='right')

    # Fill the entire region above the Planck limit with gray shading
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    xaxes = np.linspace(xmin, xmax, num=50)
    ax.fill_between(xaxes, planck_limit, ymax, color='gray', alpha=0.3)

    # --- Other settings
    # Set x-axis ticks in log scale manner with explicit scientific formatting and increase tick size
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'$10^{{{int(np.log10(val))}}}$'))
    ax.tick_params(axis='x', which='major', labelsize=12)  # Increase x-axis tick size

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # ---------------------------- RELATIVE ERROR: MAXIM RESULTS VS OUR INTEGRATION ---------------------------------- #
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot relative error as a function of axion mass
    rel_diff_arr = abs(delta_Neff_MAXIM - our_data["delta_neff"])/abs(delta_Neff_MAXIM) * 100
    plt.plot(fa_arr_MAXIM, rel_diff_arr, 'go-', label=r'Relative Error (%)', markersize=5)

    # --- Lines
    # Add a horizontal line at 1% to highlight the accuracy threshold
    ax.axhline(y=1, color='r', linestyle='--', label='1% threshold')

    # --- Labels with bold text
    ax.set_xlabel(r'Decay Constant $\bf f_a$ [GeV]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Relative Error [%]', fontsize=14, fontweight='bold')
    ax.set_title(title_arr[file_number], fontsize=16, fontweight='bold')

    # --- Other settings
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)

    plt.show()
