import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad

class AxionThermalModelDistribution:
    """
    """
    # ----------------------------------------- INITIALIZATION ------------------------------------------------------- #
    def __init__(self, axion_mass):
        """
        """
        # ### Physical Constants ##################################################################################### #
        self.con = dict()  # dictionary with constants
        self.con["rho_crit"] = 8.098 * 10**(-11)  # critical density, [h^2 eV^4] natural units

        # --- Constants corresponding to photon ---
        self.con["g_dof_photon"] = 2  # photon degrees of freedom
        # Photon temperature today (nowadays)
        self.con["T_photon"] = dict()
        self.con["T_photon"]["[K]"] = 2.7255                      # [K] in Kelvin
        self.con["T_photon"]["[eV]"] = 2.7255 * 8.617 * 10**(-5)  # kB * T, [kB*T] = [kB] * [T] = [eV/K] * [K] = [eV]
        self.con["T_photon"]["[T_photon]"] = 1                    # [T_photon]

        # --- Constants corresponding to neutrinos ---
        self.con["g_dof_nu"] = 3*2  # neutrinos degrees of freedom
        # Neutrinos temperature today (nowadays)
        self.con["T_nu"] = dict()
        self.con["T_nu"]["[K]"]  = 0.71611 * self.con["T_photon"]["[K]"]   # [K]
        self.con["T_nu"]["[eV]"] = 0.71611 * self.con["T_photon"]["[eV]"]  # [eV]
        self.con["T_nu"]["[T_photon]"] = 0.71611                           # [T_photon]

        # --- Constants corresponding to axion ---
        self.con["g_dof_axion"] = 1  # axion degrees of freedom
        self.con["ma"] = axion_mass  # [eV], axion mass
        # Axion temperature today (nowadays)
        self.con["T_a"] = dict()
        self.con["T_a"]["[K]"] = None               # [K]
        self.con["T_a"]["[eV]"] = None              # [eV]
        self.con["T_a"]["[T_photon]"] = None        # [T_photon]


    # ----------------------------------------- BASIC FUNCTION ------------------------------------------------------- #
    def get_axion_energy_density(self, Ta, infinite_accuracy=False):
        """
        Compute the present-day energy density of a thermal axion.

        Parameters:
        -----------
        - Ta (float): Present-day axion temperature in units of photon temperature (i.e., T_photon = 1), [T_photon]
        - infinite_accuracy (bool): flag to differentiate accuracy of the integration.

        Returns:
        --------
        rho_axion (float): Axion energy density in units of T_photon^4.
        """
        # Rescale mass to units of Ta
        Ta_eV = Ta * self.con["T_photon"]["[eV]"]
        x = self.con["ma"] / Ta_eV

        def integrand(p):
            return p ** 2 * np.sqrt(p ** 2 + x ** 2) / (np.exp(p) - 1)

        # Numerical integration
        if infinite_accuracy:
            integral, _ = quad(integrand, 0, np.inf, limit=300, epsabs=1e-10)
        else:
            integral, _ = quad(integrand, 0, 50, limit=300, epsabs=1e-10)

        # Axion Energy Density, [T_photon^4]
        rho_axion = self.con["g_dof_axion"] / (2 * np.pi ** 2) * Ta ** 4 * integral  # [T_photon^4]

        return rho_axion  # [T_photon^4]

    # ----------------------------------------- FIND T_a ------------------------------------------------------------- #
    def delta_neff_from_Ta(self, Ta, infinite_accuracy=False):
        """
        Calculate ΔNeff for a thermal axion with given temperature Ta and mass m_a.

        Parameters:
        -----------
        - Ta (float): Present-day axion temperature in units of photon temperature (i.e., T_photon = 1), [T_photon]
        - infinite_accuracy (bool): flag to differentiate accuracy of the integration.

        Returns:
        --------
        delta_Neff (float): The contribution of the axion to the effective number of relativistic degrees of freedom.
        """
        # Axion Energy Density, [T_photon^4]
        rho_axion = self.get_axion_energy_density(Ta, infinite_accuracy=infinite_accuracy)  # [T_photon^4]

        # Energy density of one massless neutrino species (in same units)
        rho_nu_1 = (7 / 8) * (np.pi ** 2 / 15) * self.con["T_nu"]["[T_photon]"] ** 4  # [T_photon]

        # Return ratio = ΔNeff
        return rho_axion / rho_nu_1

    def find_Ta_for_DeltaNeff(self, DeltaNeff_target, integral=False):
        """
        Solve for the axion temperature (Ta) that yields a specific ΔNeff value.

        This function inverts the BE-based energy density formula numerically
        by finding the value of Ta that gives the desired ΔNeff for a given axion mass.

        Parameters:
        -----------
        m_a_eV (float): Axion mass in [eV].
        DeltaNeff_target (float): Target contribution to ΔNeff (e.g. 0.5).
        g_a : int
            Axion internal degrees of freedom (1 for real scalar).

        Returns:
        --------
        Ta_solution : float
            The required present-day axion temperature (in units of T_gamma)
            such that the axion contributes exactly the desired ΔNeff.
        """
        if integral:
            # Define root function: find Ta where ΔNeff(Ta) - target = 0
            def f(Ta):
                return self.delta_neff_from_Ta(Ta) - DeltaNeff_target

            # Solve numerically using Brent's method between reasonable bounds
            result = root_scalar(f, bracket=[1e-4, 2.0], method='brentq')

            # Return the solution
            return result.root
        else:
            factor = 8/7 * (11/4)**(4/3) * self.con["g_dof_axion"] / self.con["g_dof_photon"]
            Ta = (DeltaNeff_target/factor) ** (1/4)  # [T_photon]

            return Ta

    # ----------------------------------------- Calculate omega_a ---------------------------------------------------- #
    def get_omega_axion(self, Ta, infinite_accuracy=False):
        """
        """
        # Axion Energy Density, [T_photon^4]
        rho_axion = self.get_axion_energy_density(Ta, infinite_accuracy=infinite_accuracy)  # [T_photon^4]
        rho_axion_eV = rho_axion * self.con["T_photon"]["[eV]"]**4  # [eV^4]

        # Find nowadays abundance of the axion
        omega_axion = rho_axion_eV  / self.con["rho_crit"]  # [h^2]
        return omega_axion

    # ----------------------------------------- NAIVE: ΔNeff --------------------------------------------------------- #
    def naive_deltaNeff(self, DeltaNeff_target, infinite_accuracy=False):
        """
        """
        # Rescale mass to units of Ta
        T_CMB = self.con["T_photon"]["[eV]"]
        x = self.con["ma"] / T_CMB

        def integrand(p):
            return p ** 2 * np.sqrt(p ** 2 + x ** 2) / (np.exp(p) - 1)

        # Numerical integration
        if infinite_accuracy:
            integral, _ = quad(integrand, 0, np.inf, limit=300, epsabs=1e-10)
        else:
            integral, _ = quad(integrand, 0, 50, limit=300, epsabs=1e-10)

        # Axion Energy Density, [T_photon^4]
        deltaNeff_CMB = (15/np.pi**4 * integral)**(1) * DeltaNeff_target
        return deltaNeff_CMB

# ####################################### CROSS CHECKS ############################################################### #
if __name__ == "__main__":
    # --- AXION: the range of parameters
    ma = 1e2  # [eV]
    Nums = 20

    min_deltaNeff = 0.01
    max_deltaNeff = 0.8
    deltaNeff_arr = np.logspace(np.log10(min_deltaNeff), np.log10(max_deltaNeff), Nums)

    # --- Create a Class
    axionModel = AxionThermalModelDistribution(ma)
    T_ncdm_arr = [axionModel.find_Ta_for_DeltaNeff(deltaNeff, integral=False) for deltaNeff in deltaNeff_arr]
    omega_axion_arr = [axionModel.get_omega_axion(T_ncdm) for T_ncdm in T_ncdm_arr]
    deltaNeffFull_arr = [axionModel.delta_neff_from_Ta(T_ncdm) for T_ncdm in T_ncdm_arr]
    deltaNeffCMB_arr = [axionModel.naive_deltaNeff(deltaNeff) for deltaNeff in deltaNeff_arr]

    # --- Printing info
    for i in range(0, Nums):
        T_ncdm_i = T_ncdm_arr[i]
        omega_axion_i = omega_axion_arr[i]
        print(f"Axion T_ncdm: {T_ncdm_i} [T_gamma], CLASS omega_a: {omega_axion_i}")

    print("")
    print("-----------------------------------------------------------------------------------------------------------")
    # --- Printing info
    for i in range(0, Nums):
        deltaNeffFull_i = deltaNeffFull_arr[i]
        deltaNeffCMB_i = deltaNeffCMB_arr[i]
        deltaNeff_i = deltaNeff_arr[i]
        print(f"Axion ΔNeff_i: {deltaNeffFull_i} [T_gamma], ΔNeffCMB_i: {deltaNeffCMB_i}, what I SET: {deltaNeff_i}")

    print("")
    print("-----------------------------------------------------------------------------------------------------------")
    # --- Printing info
    for i in range(0, Nums):
        T_ncdm_arr_i = T_ncdm_arr[i]
        factor = 8/7 * (11/4)**(4/3) * 1/2
        deltaNeff_val = factor * T_ncdm_arr_i**4
        
        print("deltaNeff_val:", deltaNeff_val)

    # -----------------------------------------------------------------------------------------------------------------
    ma = 1e-2  # [eV]
    set_deltaNeff = 0.45

    axionModel = AxionThermalModelDistribution(ma)
    T_ncdm_value = axionModel.find_Ta_for_DeltaNeff(set_deltaNeff, integral=False)
    omega_axion_value = axionModel.get_omega_axion(T_ncdm_value)
    deltaNeffFull_value = axionModel.delta_neff_from_Ta(T_ncdm_value)
    deltaNeffCMB_value = axionModel.naive_deltaNeff(set_deltaNeff)

    print()
    print("-----------------------------------------------------------------------------------------------------------")
    print("ma:", ma)
    print("T_ncdm_value:", T_ncdm_value)
    print("omega_axion_value:", omega_axion_value)
    print("deltaNeffFull_value:", deltaNeffFull_value)
    print("deltaNeffCMB_value:", deltaNeffCMB_value)
    factor = 8 / 7 * (11 / 4) ** (4 / 3) * 1 / 2
    deltaNeff_val = factor * T_ncdm_value ** 4
    print("deltaNeff_val:", deltaNeff_val)

    # -----------------------------------------------------------------------------------------------------------------
    ma = 1e1  # [eV]
    set_deltaNeff = 0.06

    axionModel = AxionThermalModelDistribution(ma)
    T_ncdm_value = axionModel.find_Ta_for_DeltaNeff(set_deltaNeff, integral=False)
    omega_axion_value = axionModel.get_omega_axion(T_ncdm_value)
    deltaNeffFull_value = axionModel.delta_neff_from_Ta(T_ncdm_value)
    deltaNeffCMB_value = axionModel.naive_deltaNeff(set_deltaNeff)

    print()
    print("-----------------------------------------------------------------------------------------------------------")
    print("ma:", ma)
    print("T_ncdm_value:", T_ncdm_value)
    print("omega_axion_value:", omega_axion_value)
    print("deltaNeffFull_value:", deltaNeffFull_value)
    print("deltaNeffCMB_value:", deltaNeffCMB_value)
    factor = 8 / 7 * (11 / 4) ** (4 / 3) * 1 / 2
    deltaNeff_val = factor * T_ncdm_value ** 4
    print("deltaNeff_val:", deltaNeff_val)



