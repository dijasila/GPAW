from ase.lattice import bulk
from gpaw import GPAW

# This test calculates a GLLB quasiparticle gap with LCAO and verifies
# that it does not change from a reference value.  Note that the
# calculation, physically speaking, is garbage.

si = bulk('Si', 'diamond', a=5.421)
calc = GPAW(mode='lcao',
            h=0.3,
            basis='sz(dzp)',
            stencils=(2, 2),
            fixdensity=True,
            xc='GLLBSC',
            kpts={'size': (2, 2, 2), 'gamma': True},
            txt='si.txt')


def stopcalc():
    calc.scf.converged = True
calc.attach(stopcalc, 1)

si.set_calculator(calc)
si.get_potential_energy()

response = calc.hamiltonian.xc.xcs['RESPONSE']
response.calculate_delta_xc()
EKs, Dxc = response.calculate_delta_xc_perturbation()
refgap = 3.01514230845
gap = EKs + Dxc
print('GAP', gap)
assert abs(gap - refgap) < 1e-4
