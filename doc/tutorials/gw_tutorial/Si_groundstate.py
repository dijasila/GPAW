from ase.structure import bulk
from gpaw import GPAW, FermiDirac

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

h = 0.20
kpts = (3,3,3)
nbands = 100

calc = GPAW(
            h=h,
            kpts=kpts,
            xc='LDA',
            txt='Si_groundstate.txt',
            nbands=nbands,
            convergence={'bands':nbands-10}, # for faster convergence
            eigensolver='cg',                # for faster convergence
            occupations=FermiDirac(0.001)    # for faster convergence
           )

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_groundstate.gpw','all')
