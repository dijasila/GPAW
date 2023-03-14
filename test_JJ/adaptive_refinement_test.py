from gpaw import GPAW, kpt_descriptor, FermiDirac
import numpy as np

calc_old = GPAW(f'gs_BrCuS_kpts50x50.gpw', txt=None)
kpts_new = [[-0.46666667, -0.26666667,  0.        ],
            [-0.46666667, -0.2,         0.        ],
            [-0.46666667, -0.33333333,  0.        ],
            [-0.46666667, -0.13333333,  0.        ],
            [-0.46666667, -0.06666667,  0.        ],
            [-0.46666667, -0.46666667,  0.        ],
            [-0.46666667, -0.4,         0.        ]]

calc_new = calc_old.fixed_density(kpts = kpts_new, nbands='200%', occupations=FermiDirac(0.05), symmetry='off', txt=f'gs_BrCuS_kpts7_convergeddensity_refined_N_3_test.txt')
calc_new.write(f'gs_BrCuS_kpts7_convergeddensity_refined_N_3_test.gpw', mode='all')    

kpts_new2 = [[-0.46666667, -0.46666667,  0.        ],
            [-0.46666667, -0.4,         0.        ],
            [-0.46666667, -0.33333333,  0.        ],
            [-0.46666667, -0.26666667,  0.        ],
            [-0.46666667, -0.2,         0.        ],
            [-0.46666667, -0.13333333,  0.        ],
            [-0.46666667, -0.06666667,  0.        ]]

calc1 = GPAW(f'gs_BrCuS_kpts50x50.gpw', txt=None)
calc_new2 = calc1.fixed_density(kpts = kpts_new2, nbands='200%', occupations=FermiDirac(0.05), symmetry='off', txt=f'gs_BrCuS_kpts7_convergeddensity_test.txt')
calc_new2.write(f'gs_BrCuS_kpts7_convergeddensity_test.gpw', mode='all')





