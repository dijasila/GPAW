from ase.build import bulk
from gpaw import GPAW, PW, FD, FermiDirac
from ase.collections import g2
from gpaw.directmin.fdpw.directmin import DirectMin
from gpaw.mpi import world
from ase.dft.bandgap import bandgap
from ase import io

atoms = bulk('C', a=3.572466, crystalstructure='diamond', cubic=True)
atoms = atoms.repeat((2, 2, 2))
calc = GPAW(xc='PBE',
            mode=PW(),
            eigensolver=DirectMin(
                searchdir_algo={'name': 'LBFGS', 'memory': 1},
            ),
            occupations={'name': 'fixed-occ-zero-width'},
            mixer={'name': 'dummy'},
            convergence={'eigenstates':1.0e-10},
            )
atoms.calc = calc
atoms.get_potential_energy()
