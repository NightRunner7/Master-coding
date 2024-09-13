import numpy as np
import matplotlib.pyplot as plt
from read_data_muon_scattering import DistributionOfMuonScattering

file_path = './Maxim-muon-scattering/ma_distributions_mu_scattering.dat'  # Set the correct file path
analyzer = DistributionOfMuonScattering(file_path)

# --- Do all necessary stuff to get the fit of parameters
analyzer.find_all_fit_distribution_parameters()  # find all parameters
analyzer.find_polynomial_fit_of_parameters()  # find all polynomials

# --- Take all necessary data from analyzer
# axion mass
axion_masses = analyzer.return_axion_masses()
log_masses = np.log(axion_masses)

# all parameters: A, b, mu
fit_parameters = analyzer.return_fit_parameters()

# all polynomials
polynomial_fit = analyzer.return_polynomial_of_parameters()


# ------------------------------------------------------- PLOTS ------------------------------------------------------ #
# --- PLOT OF PARAMETER A
plt.figure(figsize=(10, 6))
# DATA
plt.scatter(log_masses, fit_parameters['A'],
            color='black', label=r'Param $A$ (Approx. Dist.)', alpha=0.7, s=50)
sorted_indexes = np.argsort(log_masses)
plt.plot(log_masses[sorted_indexes], polynomial_fit['A'](log_masses[sorted_indexes]),
         label='Fitted Polynomial', color='red', linewidth=2)
# DESCRIPTION
plt.xlabel(r'$\bf{log(m_{a})}$ [$ \bf{log(eV)}$]', fontsize=14, fontweight='bold')
plt.ylabel(r'Parameter: $\bf A(m_{a})$', fontsize=14, fontweight='bold')
# plt.title(r'Fitting Polynomial to Parameter $\bf A$ vs. $\bf log(m_{a})$', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('Fitted_Parameters_Plot_A.png', format='png', dpi=300)  # Save as PNG with high resolution
# plt.show()


# --- PLOT OF PARAMETER b
plt.figure(figsize=(10, 6))
# DATA
plt.scatter(log_masses, fit_parameters['b'],
            color='black', label=r'Param $b$ (Approx. Dist.)', alpha=0.7, s=50)
sorted_indexes = np.argsort(log_masses)
plt.plot(log_masses[sorted_indexes], polynomial_fit['b'](log_masses[sorted_indexes]),
         label='Fitted Polynomial', color='red', linewidth=2)
# DESCRIPTION
plt.xlabel(r'$\bf{log(m_{a})}$ [$ \bf{log(eV)}$]', fontsize=14, fontweight='bold')
plt.ylabel(r'Parameter: $\bf b(m_{a})$', fontsize=14, fontweight='bold')
# plt.title(r'Fitting Polynomial to Parameter $\bf b$ vs. $\bf log(m_{a})$', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('Fitted_Parameters_Plot_b.png', format='png', dpi=300)  # Save as PNG with high resolution
# plt.show()

# --- PLOT OF PARAMETER mu
plt.figure(figsize=(10, 6))
# DATA
plt.scatter(log_masses, fit_parameters['mu'],
            color='black', label=r'Param $\mu$ (Approx. Dist.)', alpha=0.7, s=50)
sorted_indexes = np.argsort(log_masses)
plt.plot(log_masses[sorted_indexes], polynomial_fit['mu'](log_masses[sorted_indexes]),
         label='Fitted Polynomial', color='red', linewidth=2)
# DESCRIPTION
plt.xlabel(r'$\bf{log(m_{a})}$ [$ \bf{log(eV)}$]', fontsize=14, fontweight='bold')
plt.ylabel(r'Parameter: $\bf \mu(m_{a})$', fontsize=14, fontweight='bold')
# plt.title(r'Fitting Polynomial to Parameter $\bf \mu$ vs. $\bf log(m_{a})$', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('Fitted_Parameters_Plot_mu.png', format='png', dpi=300)  # Save as PNG with high resolution
# plt.show()




# # Create a figure and a set of subplots
# fig, axs = plt.subplots(2, 2, figsize=(16, 9))
#
# # Remove the empty subplot (top right)
# fig.delaxes(axs[0][1])
#
# # Plot for Parameter A
# ax = axs[0][0]
# ax.scatter(log_masses, fit_parameters['A'], color='black', alpha=0.7, s=50, label='Param $A$ (Approx. Dist.)')
# ax.plot(log_masses, polynomial_fit['A'](log_masses), 'r-', linewidth=2, label='Fitted Polynomial')
# ax.set_xlabel(r'$\bf{log(m_{a})}$ [$ \bf{log(eV)}$]', fontsize=14, fontweight='bold')
# ax.set_ylabel(r'Parameter: $\bf b(m_{a})$', fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(True, linestyle='--', linewidth=0.5)
#
# # Plot for Parameter b
# ax = axs[1][0]
# ax.scatter(log_masses, fit_parameters['b'], color='black', alpha=0.7, s=50, label='Param $b$ (Approx. Dist.)')
# ax.plot(log_masses, polynomial_fit['b'](log_masses), 'r-', linewidth=2, label='Fitted Polynomial')
# ax.set_xlabel(r'$\bf{log(m_{a})}$ [$ \bf{log(eV)}$]', fontsize=14, fontweight='bold')
# ax.set_ylabel(r'Parameter: $\bf b(m_{a})$', fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(True, linestyle='--', linewidth=0.5)
#
# # Plot for Parameter mu
# ax = axs[1][1]
# ax.scatter(log_masses, fit_parameters['mu'], color='black', alpha=0.7, s=50, label='Param $\mu$ (Approx. Dist.)')
# ax.plot(log_masses, polynomial_fit['mu'](log_masses), 'r-', linewidth=2, label='Fitted Polynomial')
# ax.set_xlabel(r'$\bf{log(m_{a})}$ [$ \bf{log(eV)}$]', fontsize=14, fontweight='bold')
# ax.set_ylabel(r'Parameter: $\bf \mu(m_{a})$', fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(True, linestyle='--', linewidth=0.5)
#
# # Adjust layout to avoid overlap and ensure visibility
# plt.tight_layout()
# # plt.subplots_adjust(top=1.28)
# plt.show()

