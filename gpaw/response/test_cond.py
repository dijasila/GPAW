from gpaw.response.MPAinterpolation import mpa_RE_solver
import numpy as np

omega = np.array([0.+3.67493225e-12j, 0.+1.00000000e+00j]) 
einv = np.array([7.26813572e-10+8.02893598e-10j, 3.11863353e-10+1.16709471e-09j])

omegat_n, R_n, MPred, PPcond_rate = mpa_RE_solver(1, omega, einv)

print('omega ', omegat_n, 'Res ', R_n, 'cond ', PPcond_rate)
