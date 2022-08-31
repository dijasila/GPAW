from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW, PW
from ase.constraints import UnitCellFilter


a = Atoms('BaTiO3',
          cell=[3.98, 3.98, 4.07],
          pbc=True,
          scaled_positions=[[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.8],
                            [0.5, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.5]])

a.calc = GPAW(mode=PW(800),
              xc='PBE',
              kpts={'size': (6, 6, 6), 'gamma': True},
              symmetry='off',
              txt='relax.txt')

uf = UnitCellFilter(a, mask=[1, 1, 1, 0, 0, 0])
opt = BFGS(uf)
opt.run(fmax=0.01)
a.calc.write('BaTiO3.gpw', mode='all')
