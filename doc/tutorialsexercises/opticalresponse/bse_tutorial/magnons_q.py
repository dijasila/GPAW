import numpy as np
from gpaw.response.bse import BSE
from ase.parallel import paropen
from ase.units import Ha

cutoff = 300

qs_qc = [[0.5, 0, 0],
         [0.375, 0, 0],
         [0.25, 0, 0],
         [0.125, 0, 0],
         [0, 0, 0],
         [0, 0.25, 0],
         [0, 0.5, 0]]

for i, q_c in enumerate(qs_qc):
    bse = BSE('gs_RhCl2.gpw',
              spinors=True,
              ecut=cutoff,
              valence_bands=[54, 55, 56, 57],
              conduction_bands=[58, 59, 60, 61, 62, 63],
              eshift=2.4,
              nbands=40,
              truncation='2D',
              q_c=q_c,
              wfile='W_RhCl2_%s' % cutoff,
              mode='BSE',
              txt='bse_RhCl2_q%s.txt' % i)

    bse.get_magnetic_susceptibility(pbc=[True, True, False],
                                    eta=0.1,
                                    susc_component='+-',
                                    w_w=np.linspace(0, 1, 100))

    fd = paropen('magnons_q.dat', 'a')
    if i < 4:
        q = -2 * np.pi * np.dot(q_c, q_c)**0.5 / 3.5006
    else:
        q = 2 * np.pi * np.dot(q_c, q_c)**0.5 / 6.8529        
    print(q, bse.w_T[0] * Ha, bse.w_T[1] * Ha, file=fd)
    fd.close()
