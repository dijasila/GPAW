# web-page: unocc.out
# Start
from gpaw import GPAW

calc = GPAW('gs.gpw')
calc = calc.fixed_density(nbands=42,
                          convergence={'bands': 40},
                          maxiter=1000,
                          txt='unocc.out')
calc.write('unocc.gpw', mode='all')
