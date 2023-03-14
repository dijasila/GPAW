from gpaw.response.berryology import get_hall_conductivity
import numpy as np


calc_file = f'gs_BrCuS_kpts7_convergeddensity_test.gpw'
sigma_xy, sigma_yz, sigma_zx = get_hall_conductivity(calc_file)
print(sigma_xy)
np.savetxt(f'BrCuS_sigmaxy_kpts7_convergeddensity_test.npy', sigma_xy)


calc_file = f'gs_BrCuS_kpts7_convergeddensity_refined_N_3_test.gpw'
sigma_xy, sigma_yz, sigma_zx = get_hall_conductivity(calc_file)
print(sigma_xy)
np.savetxt(f'BrCuS_sigmaxy_kpts7_convergeddensity_refined_N_3_test.npy', sigma_xy)
