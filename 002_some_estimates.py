import numpy as np
from scipy.special import erf

std_da = 0.2

def cdf(r):
    return 0.5 * (1 + erf(r / np.sqrt(2)))

def cdf_da(da):
    return cdf(da / std_da)

def cdf_diff_da_2_runs(da):
    return cdf(da / np.sqrt(2) / std_da)

# probability of getting da of by 0.2
p_da_0_2 = 1 - (cdf_da(0.2) - cdf_da(-0.2))
print(f'Probability of getting da off by 0.2: {p_da_0_2*100:.2f} %')

# probability of getting da of by 0.5
p_da_0_5 = 1 - (cdf_da(0.5) - cdf_da(-0.5))
print(f'Probability of getting da off by 0.5: {p_da_0_5*100:.2f} %')

# probability of getting a difference of 1 sigma in da between 2 runs
p_diff_da_1_sigma = 1 - (cdf_diff_da_2_runs(1) - cdf_diff_da_2_runs(-1))
print(f'Probability of getting a difference of 1 sigma in da between 2 runs: {p_diff_da_1_sigma*100:.2f} %')

# probability of getting a difference of 1 sigma over a scan with n runs
n_runs = 900
p_diff_da_1_sigma_scan = 1 - (1 - p_diff_da_1_sigma)**(n_runs)
print(f'Probability of getting a difference of 1 sigma over a scan with {n_runs} runs: {p_diff_da_1_sigma_scan*100:.2f} %')



