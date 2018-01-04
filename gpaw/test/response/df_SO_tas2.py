from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.response.chi0_SO import DielectricFunctionSO
from gpaw.response.df import DielectricFunction
import numpy as np
from gpaw.mpi import world


atoms = Atoms('TaS2',
              positions=[[0.00000000, 0.00000000, 9.06471960],
                         [1.67046180, 0.96444157, 10.63051065],
                         [1.67046180, 0.96444157, 7.49892856]],
              cell=[[3.34092360506, 0.0, 0.0],
                    [-1.67046180253, 2.89332471409, 0.0],
                    [0.0, 0.0, 18.1294392107]],
              pbc=[True, True, False])

atoms = Atoms('TaS2',
              positions=[[0.00000000, 0.00000000, 0.0],
                         [1.67046180, 0.96444157, 10.63051065 - 9.06471960],
                         [1.67046180, 0.96444157, 7.49892856 - 9.06471960]],
              cell=[[3.34092360506, 0.0, 0.0],
                    [-1.67046180253, 2.89332471409, 0.0],
                    [0.0, 0.0, 6.05]],
              pbc=[True, True, True])


# calc = GPAW(mode='pw',
#             kpts={'density': 5})

# atoms.calc = calc
# atoms.get_potential_energy()
# calc.write('df_so_gs.gpw')

# nval = calc.wfs.nvalence

# escalc = GPAW('df_so_gs.gpw',
#               fixdensity=True,
#               kpts={'size': (24, 24, 8), 'gamma': True},
#               nbands=2 * nval,
#               convergence={'bands': 1 * nval},
#               occupations=FermiDirac(width=1e-4))
# escalc.get_potential_energy()
# escalc.write('df_so_es.gpw', 'all')

kwargs = {'eta': 0.05,
          'domega0': 0.005,
          'integrationmode': 'tetrahedron integration',
          'ecut': 10,
          'intraband': True,
          'nblocks': 1}


df = DielectricFunctionSO('df_so_es.gpw',
                          **kwargs)
alpha0x, alphax = df.get_polarizability(q_c=[0, 0, 0],
                                        direction='x',
                                        filename=None)
frequencies = df.get_frequencies()
data = {'alpha0x': np.array(alpha0x),
        'alphax': np.array(alphax),
        'frequencies': frequencies}

filename = 'pol_tas2.npz'

if world.rank == 0:
    np.savez_compressed(filename, **data)
