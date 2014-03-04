from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.mixer import ExperimentalDotProd
from gpaw.test import equal
#from gpaw.atom.generator import Generator
from gpaw.test import gen

#Test tf_mode for H and C

h = 0.3
a = 8
c = a/2
d = 1.8

elements = ['C']
results = [1.59042607826]
electrons = [6]


for symbol in elements:
	xcname = 'PXC:1.0*LDA_K_TF+1.0*LDA_X'
	g = gen(symbol, xcname=xcname,scalarrel=False,tf_mode=True)

for element, result,e in zip(elements, results,electrons):
	atom = Atoms(element,
			positions = [(c,c,c)],
			cell = (a,a,a))

	mixer = Mixer(0.3, 5, 1)
	#mixer=Mixer(beta=0.05, nmaxold=2, weight=50.0)
	calc = GPAW(h=h,nbands=1, txt='-', xc=xcname, maxiter=240, 
		    eigensolver='cg',mixer=mixer, tf_mode=True)
	#mixer.dotprod = ExperimentalDotProd(calc)

	atom.set_calculator(calc)

	E = atom.get_total_energy()
	n = calc.get_all_electron_density()

	dv = atom.get_volume() / calc.get_number_of_grid_points().prod()
	I = n.sum() * dv / 2**3

	equal(result, E, 1.0e-4)
	equal(I, e, 1.0e-6)
