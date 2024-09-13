import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from read_data_muon_scattering import DistributionOfMuonScattering

# -------------------------------------- HELPFULLY FUNCTION ---------------------------------------------------------- #
def log_space(start, end, n):
    """Generate logarithmically spaced points."""
    return np.exp(np.linspace(np.log(start), np.log(end), n))


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


def f_boltzmann(q, C):
    """Boltzmann's distribution function, where C is some constant"""
    return C * np.exp(-q)


def f_boltzmann_times_q_squared(q, C):
    """Boltzmann's distribution function weighted by q^2, where C is some constant"""
    return C * np.exp(-q) * q ** 2


def f_boltzmann_times_q_cubic(q, C):
    """Boltzmann's distribution function weighted by q^3, where C is some constant"""
    return C * np.exp(-q) * q ** 3


def f_einstein(q, C):
    """Einstein-Bose distributions function, where C is some constant"""
    return C * (np.exp(q) - 1) ** (-1)


def f_einstein_times_q_squared(q, C):
    """Einstein-Bose distributions function weighted by q^2, where C is some constant"""
    return C * (np.exp(q) - 1) ** (-1) * q ** 3


def f_einstein_times_q_cubic(q, C):
    """Einstein-Bose distributions function weighted by q^3, where C is some constant"""
    return C * (np.exp(q) - 1) ** (-1) * q ** 3


# -------------------------------------- SET UP ---------------------------------------------------------------------- #
file_path = './Maxim-muon-scattering/ma_distributions_mu_scattering.dat'  # Set the correct file path
select_index = 10
analyzer = DistributionOfMuonScattering(file_path)

# --- Do all necessary stuff to get the fit of parameters
analyzer.find_all_fit_distribution_parameters()  # find all parameters
analyzer.find_polynomial_fit_of_parameters()  # find all polynomials

# --- Take all necessary data from analyzer
# axion mass
axion_masses = analyzer.return_axion_masses()
selected_mass = axion_masses[select_index]
print("selected_mass:", selected_mass, "[eV]")

# Real distribution f(q) * q^2 (Maxim)
q_data_arr = analyzer.return_comoving_momenta()  # [dimensionless]
distributions_weighted_q_squared = analyzer.distributions_weighted_q_squared[select_index]

# distribution first interpolation
distributions_weighted_q_squared_firstInter = interp1d(q_data_arr, distributions_weighted_q_squared, kind='cubic')

# Distribution from second interpolation
q_arr = log_space(1e-4, 20.0, 700)  # [dimensionless]
fit_parameters = analyzer.return_fit_parameters()
fit_parameter_A = fit_parameters['A'][select_index]
fit_parameter_b = fit_parameters['b'][select_index]
fit_parameter_mu = fit_parameters['mu'][select_index]

distributions_weighted_q_squared_secondInter = \
    [f_approx(q, fit_parameter_A, fit_parameter_b, fit_parameter_mu) * q ** (-1) for q in q_arr]
# print("fit_parameter_A:", fit_parameter_A)
# print("fit_parameter_b:", fit_parameter_b)
# print("fit_parameter_mu:", fit_parameter_mu)

# -------------------------------------- MAKE PLOTS ------------------------------------------------------------------ #
# --- Original vs Interpolated Distribution
plt.figure(figsize=(10, 6))
# Original data points
plt.scatter(q_data_arr, distributions_weighted_q_squared, color='black',
            label='Original Distribution ($q^2 \cdot f(q)$)', alpha=0.7, s=50)
# Interpolated function
plt.plot(q_arr, distributions_weighted_q_squared_firstInter(q_arr), color='red',
         label='Interpolated Distribution ($q^2 \cdot f(q)$)', linewidth=2)

# Enhancing descriptions
plt.xlabel(r'Comoving Momentum $\bf q$ [dimensionless]', fontsize=14, fontweight='bold')
plt.ylabel(r'Distribution Weighted by $\bf q^2$, $\bf q^2 \cdot f(q)$', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('Original_vs_Interpolated_first_Distribution.png', format='png', dpi=300)
# plt.show()


# --- Original vs Interpolated Distribution (second interpolation)
plt.figure(figsize=(10, 6))
# Original data points
plt.scatter(q_data_arr, distributions_weighted_q_squared, color='black',
            label='Original Distribution ($q^2 \cdot f(q)$)', alpha=0.7, s=50)
# Interpolated function
plt.plot(q_arr, distributions_weighted_q_squared_secondInter, color='red',
         label='Interpolated Distribution ($q^2 \cdot f(q)$)', linewidth=2)

# Enhancing descriptions
plt.xlabel(r'Comoving Momentum $\bf q$ [dimensionless]', fontsize=14, fontweight='bold')
plt.ylabel(r'Distribution Weighted by $\bf q^2$, $\bf q^2 \cdot f(q)$', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('Original_vs_Interpolated_Distribution.png', format='png', dpi=300)  # Save as PNG with high resolution


# plt.show()

# --- Interpolated vs Equilibrium Distribution, weighted by q^3
def interpolated_dis_to_integrate(q):
    return f_approx(q, fit_parameter_A, fit_parameter_b, fit_parameter_mu)


# new_f_approx = analyzer.interpolation_dist_functions[select_index]
# Compute the integral of the selected distribution
upperLimit = 19
integral_of_interpolated_dist, _ = quad(interpolated_dis_to_integrate, 0, upperLimit)

print("integralF:", integral_of_interpolated_dist)


# Define the integral functions
def integralFBoltzmann(C):
    return quad(f_boltzmann_times_q_cubic, 0, upperLimit, args=(C,))[0]


def integralFEinstein(C):
    return quad(f_einstein_times_q_cubic, 0, upperLimit, args=(C,))[0]


# Function to find differences from integralF
def diffBoltzmann(C):
    return integralFBoltzmann(C) - integral_of_interpolated_dist


def diffEinstein(C):
    return integralFEinstein(C) - integral_of_interpolated_dist


# Solve for C using root finding
solBoltzmann = root_scalar(diffBoltzmann, bracket=[0, 10], method='brentq')
solEinstein = root_scalar(diffEinstein, bracket=[0, 10], method='brentq')

print("Solution for C in Boltzmann distribution:", solBoltzmann.root)
print("Solution for C in Einstein distribution:", solEinstein.root)

# Verify the solutions
verifiedBoltzmann = integralFBoltzmann(solBoltzmann.root)
verifiedEinstein = integralFEinstein(solEinstein.root)
print("Verified integral for Boltzmann:", verifiedBoltzmann)
print("Verified integral for Einstein:", verifiedEinstein)

# # --- PLOT
plt.figure(figsize=(12, 7))
# Interpolated function
plt.plot(q_arr, interpolated_dis_to_integrate(q_arr),
         'r-', label=r'Interpolated Distribution ($q^3 \cdot f(q)$)', linewidth=2)
# Boltzmann Distribution with the solution C from Boltzmann fit
plt.plot(q_arr, f_boltzmann_times_q_cubic(q_arr, solBoltzmann.root),
         'b--', label=r'Maxwell-Boltzmann Distribution ($q^3 \cdot f(q)$)', linewidth=2)
# Einstein Distribution with the solution C from Einstein fit
plt.plot(q_arr, f_boltzmann_times_q_cubic(q_arr, solEinstein.root),
         'k-.', label=r'Bose-Einstein Distribution ($q^3 \cdot f(q)$)', linewidth=2)

# Adding descriptions
plt.xlabel(r'Comoving Momentum $\bf q$ (dimensionless)', fontsize=14, fontweight='bold')
plt.ylabel(r'Distribution Weighted by $\bf q^3$, $\bf q^3 \cdot f(q)$', fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
# Save the plot
plt.savefig('Interpolated_vs_Equilibrium_Distribution_with_q_cubic.png', format='png', dpi=300)


# plt.show()

# --- Interpolated vs Equilibrium Distribution, weighted by q^2
def interpolated_dis_to_integrate(q):
    return f_approx(q, fit_parameter_A, fit_parameter_b, fit_parameter_mu) * q ** (-1)


# new_f_approx = analyzer.interpolation_dist_functions[select_index]
# Compute the integral of the selected distribution
upperLimit = 19
integral_of_interpolated_dist, _ = quad(interpolated_dis_to_integrate, 0, upperLimit)

print("integralF:", integral_of_interpolated_dist)


# Define the integral functions
def integralFBoltzmann(C):
    return quad(f_boltzmann_times_q_squared, 0, upperLimit, args=(C,))[0]


def integralFEinstein(C):
    return quad(f_einstein_times_q_squared, 0, upperLimit, args=(C,))[0]


# Function to find differences from integralF
def diffBoltzmann(C):
    return integralFBoltzmann(C) - integral_of_interpolated_dist


def diffEinstein(C):
    return integralFEinstein(C) - integral_of_interpolated_dist


# Solve for C using root finding
solBoltzmann = root_scalar(diffBoltzmann, bracket=[0, 10], method='brentq')
solEinstein = root_scalar(diffEinstein, bracket=[0, 10], method='brentq')

print("Solution for C in Boltzmann distribution:", solBoltzmann.root)
print("Solution for C in Einstein distribution:", solEinstein.root)

# Verify the solutions
verifiedBoltzmann = integralFBoltzmann(solBoltzmann.root)
verifiedEinstein = integralFEinstein(solEinstein.root)
print("Verified integral for Boltzmann:", verifiedBoltzmann)
print("Verified integral for Einstein:", verifiedEinstein)

# # --- PLOT
plt.figure(figsize=(12, 7))
# Interpolated function
plt.plot(q_arr, interpolated_dis_to_integrate(q_arr),
         'r-', label=r'Interpolated Distribution ($q^2 \cdot f(q)$)', linewidth=2)
# Boltzmann Distribution with the solution C from Boltzmann fit
plt.plot(q_arr, f_boltzmann_times_q_squared(q_arr, solBoltzmann.root),
         'b--', label=r'Maxwell-Boltzmann Distribution ($q^2 \cdot f(q)$)', linewidth=2)
# Einstein Distribution with the solution C from Einstein fit
plt.plot(q_arr, f_einstein_times_q_squared(q_arr, solEinstein.root),
         'k-.', label=r'Bose-Einstein Distribution ($q^2 \cdot f(q)$)', linewidth=2)

# Adding descriptions
plt.xlabel(r'Comoving Momentum $\bf q$ (dimensionless)', fontsize=14, fontweight='bold')
plt.ylabel(r'Distribution Weighted by $\bf q^2$, $\bf q^2 \cdot f(q)$', fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
# Save the plot
plt.savefig('Interpolated_vs_Equilibrium_Distribution_with_q_square.png', format='png', dpi=300)
# plt.show()
