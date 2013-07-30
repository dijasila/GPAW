from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.mixer import ExperimentalDotProd
from gpaw.test import equal

#Test tf_mode for H and C, assuming setups generated with 600 or more gpts.

h = 0.1
a = 8
c = a/2
d = 1.8

elements = ['H', 'C']
results = [0.082795, -0.002574]
electrons = [1, 6]

for element, result,e in zip(elements, results,electrons):
	atom = Atoms(element,
			positions = [(c,c,c)],
			cell = (a,a,a))

	mixer = Mixer(0.3, 5, 1)
	#mixer=Mixer(beta=0.05, nmaxold=2, weight=50.0)
	calc = GPAW(h=h,nbands=1, txt='-', xc='LDA_K_TF+LDA_X', maxiter=240, eigensolver='cg',mixer=mixer, tf_mode=True)
	#mixer.dotprod = ExperimentalDotProd(calc)

	atom.set_calculator(calc)

	E = atom.get_total_energy()
	n = calc.get_all_electron_density()

	dv = atom.get_volume() / calc.get_number_of_grid_points().prod()
	I = n.sum() * dv / 2**3

	equal(result, E, 1.0e-4)
	equal(I, e, 1.0e-6)