import numpy as np

from read_data_muon_scattering import DistributionOfMuonScattering

# -------------------------------------- SET UP ---------------------------------------------------------------------- #
file_path = './Maxim-muon-scattering/ma_distributions_mu_scattering.dat'  # Set the correct file path
select_index = 10
analyzer = DistributionOfMuonScattering(file_path)

# --- Do all necessary stuff to get the fit of parameters
analyzer.find_all_fit_distribution_parameters()  # find all parameters
analyzer.find_polynomial_fit_of_parameters()  # find all polynomials

# get the polynomials
polynomials = analyzer.return_polynomial_of_parameters()

print("Polynomial for A parameter:")
print(polynomials['A'])
print()

print("Polynomial for b parameter:")
print(polynomials['b'])
print()

print("Polynomial for mu parameter:")
print(polynomials['mu'])
print()

# check whether is properly calculated:
m_a = 0.19
val_param_A = polynomials['A'](np.log(m_a))
val_param_b = polynomials['b'](np.log(m_a))
val_param_mu = polynomials['mu'](np.log(m_a))
print("param A:", val_param_A)
print("param b:", val_param_b)
print("param mu:", val_param_mu)
