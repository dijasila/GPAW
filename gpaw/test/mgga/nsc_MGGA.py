from __future__ import print_function
from ase import Atoms
from gpaw import GPAW, Davidson, Mixer
from gpaw.test import equal

# ??? g = Generator('H', 'TPSS', scalarrel=True, nofiles=True)

atoms = Atoms('H', magmoms=[1], pbc=True)
atoms.center(vacuum=2.2)
calc = GPAW(gpts=(20, 20, 20), nbands=1, xc='oldPBE', txt='Hnsc.txt',
            mixer=Mixer(0.8),
            eigensolver=Davidson(6))
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()
niter1 = calc.get_number_of_iterations()
e1ref = calc.get_reference_energy()
de12t = calc.get_xc_difference('TPSS')
de12m = calc.get_xc_difference('M06-L')
de12r = calc.get_xc_difference('revTPSS')


print('================')
print('e1 = ', e1)
print('de12t = ', de12t)
print('de12m = ', de12m)
print('de12r = ', de12r)
print('tpss = ', e1 + de12t)
print('m06l = ', e1 + de12m)
print('revtpss = ', e1 + de12r)
print('================')

equal(e1 + de12t, -1.28881170342, 0.005)
equal(e1 + de12m, -1.24485467819, 0.005)
equal(e1 + de12r, -1.27503567927, 0.005)

# ??? g = Generator('He', 'TPSS', scalarrel=True, nofiles=True)

atomsHe = Atoms('He', pbc=True)
atomsHe.center(vacuum=2.5)
calc = GPAW(gpts=(24, 24, 24), nbands=1, xc='oldPBE', txt='Hensc.txt',
            eigensolver=Davidson(6),
            mixer=Mixer(0.8))
atomsHe.set_calculator(calc)
e1He = atomsHe.get_potential_energy()
niter_1He = calc.get_number_of_iterations()
e1refHe = calc.get_reference_energy()
de12tHe = calc.get_xc_difference('TPSS')
de12mHe = calc.get_xc_difference('M06-L')
de12rHe = calc.get_xc_difference('revTPSS')

print('================')
print('e1He = ', e1He)
print('de12tHe = ', de12tHe)
print('de12mHe = ', de12mHe)
print('de12rHe = ', de12rHe)
print('tpss = ', e1He + de12tHe)
print('m06l = ', e1He + de12mHe)
print('revtpss = ', e1He + de12rHe)
print('================')

equal(e1He + de12tHe, -0.43033976854, 0.005)
equal(e1He + de12mHe, -0.505250941459, 0.005)
equal(e1He + de12rHe, -0.485316766611, 0.005)

energy_tolerance = 0.0005
niter_tolerance = 0
equal(e1, -1.31588284981, energy_tolerance)
equal(e1He, -0.00815663578869, energy_tolerance)
