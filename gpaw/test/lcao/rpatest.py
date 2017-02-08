import numpy as np
from ase.structure import graphene_nanoribbon
from gpaw import GPAW, FermiDirac
from gpaw.lcao.rpa import calculate

# Generate a graphene orthogonal unit cell
atoms = graphene_nanoribbon(n=2, m=1, sheet=True,
                            vacuum=2.0)
atoms.set_pbc(True)

calc = GPAW(basis='dzp', mode='lcao', kpts=[1,1,1],
            occupations=FermiDirac(0.01),
            h=0.25)
atoms.set_calculator(calc)
atoms.get_forces()

epsilon_qvsw, omega_w = calculate(calc, filename=None, omegamax=5,
                                  eta=0.1, Domega=0.025, omegamin=0.0,
                                  verbose=False, cutocc=1e-5, singlet=False,
                                  HilbertTransform=False, transitions=False)

print epsilon_qvsw

# Integrate functions against five different Gaussians.  The result is a
# "function hash value" which will likely change if the calculated function
# changes in any way.
#
# TODO: Use this for a test which does not depend on a selfconsistent
# calculation so the result is only affected by the LCAO RPA code.
def fhash(omega_w, epsilon_w):
    a = 400.
    domega = omega_w[1] - omega_w[0]
    integrals = []
    for alpha in range(1, 6):
        func = a * np.exp(-4 * (omega_w - alpha)**2)
        integrals.append((func * omega_w).sum() * domega)
        #pl.plot(omega_w, func)
    return integrals

# Calculated from a single reference run of this script
ref_integrals = np.array([354.57706427906015, 708.98154052136306, 1063.4722937204776, 1413.2581042912402, 811.24776399752886])

integrals = fhash(omega_w, epsilon_qvsw[0, 0])
print('results')
print(integrals)
print('reference')
print(ref_integrals)

assert np.abs(ref_integrals - integrals).max() < 1e-2

#pl.show()

