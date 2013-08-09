import numpy as np
import numpy.random as ra
from gpaw.setup import create_setup
from gpaw.xc import XC
from gpaw.test import equal
from gpaw.atom.generator2 import generate

if 1:
    for functional in [
        'LDA_X', 'LDA_X+LDA_C_PW', 'LDA_X+LDA_C_VWN', 'LDA_X+LDA_C_PZ',
        'GGA_X_PBE+GGA_C_PBE', 'GGA_X_PBE_R+GGA_C_PBE',
        'GGA_X_B88+GGA_C_P86', 'GGA_X_B88+GGA_C_LYP',
        'GGA_X_FT97_A+GGA_C_LYP'
        ]:
        generate(['N', '-f', functional, '-w'])
        
libxc_set = [
    'LDA_X', 'LDA_X+LDA_C_PW', 'LDA_X+LDA_C_VWN', 'LDA_X+LDA_C_PZ',
    'GGA_X_PBE+GGA_C_PBE', 'GGA_X_PBE_R+GGA_C_PBE',
    'GGA_X_B88+GGA_C_P86', 'GGA_X_B88+GGA_C_LYP',
    'GGA_X_FT97_A+GGA_C_LYP'
    ]

x = 0.000001
for xcname in libxc_set:
    ra.seed(8)
    xc = XC(xcname)
    s = create_setup('N', xc)
    ni = s.ni
    nii = ni * (ni + 1) // 2
    D_p = 0.1 * ra.random(nii) + 0.4
    H_p = np.zeros(nii)

    E1 = xc.calculate_paw_correction(s, D_p.reshape(1, -1), H_p.reshape(1, -1))
    dD_p = x * ra.random(nii)
    D_p += dD_p
    dE = np.dot(H_p, dD_p) / x
    E2 = xc.calculate_paw_correction(s, D_p.reshape(1, -1))
    print xcname, dE, (E2 - E1) / x
    equal(dE, (E2 - E1) / x, 0.003)

    E2s = xc.calculate_paw_correction(s,
        np.array([0.5 * D_p, 0.5 * D_p]), np.array([H_p, H_p]))
    print E2, E2s
    equal(E2, E2s, 1.0e-12)

    D_sp = 0.1 * ra.random((2, nii)) + 0.2
    H_sp = np.zeros((2, nii))

    E1 = xc.calculate_paw_correction(s, D_sp, H_sp)
    dD_sp = x * ra.random((2, nii))
    D_sp += dD_sp
    dE = np.dot(H_sp.ravel(), dD_sp.ravel()) / x
    E2 = xc.calculate_paw_correction(s, D_sp, H_sp)
    print dE, (E2 - E1) / x
    equal(dE, (E2 - E1) / x, 0.005)
