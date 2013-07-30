import numpy as np
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import paropen

from gpaw import GPAW, PW
from gpaw.mpi import size, rank, world, serial_comm
from gpaw.xc.tools import vxc
from gpaw.xc.hybridg import HybridXC


mgo = bulk('MgO', 'rocksalt', a=4.189)
comm = world.new_communicator(np.arange(min(3, size)))
if rank < 3:
    if 1:
        mgo.calc = GPAW(mode=PW(500),
                        parallel=dict(band=1),
                        idiotproof=False,
                        communicator=comm,
                        setups={'Mg': '2e'},
                        convergence={'eigenstates': 5.e-9},
                        kpts=monkhorst_pack((2, 2, 2)) + 0.25)
        mgo.get_potential_energy()
        mgo.calc.write('mgo', 'all')

    for name in ['PBE0', 'HSE03', 'HSE06']:
        calc = GPAW('mgo', setups={'Mg': '2e'},
                   txt=None, communicator=serial_comm)
        hyb_calc = HybridXC(name, alpha=5.0, bandstructure=True, world=comm)
        de_skn = vxc(calc, hyb_calc) - vxc(calc, 'LDA')
        
        if name == 'PBE0':
            de_skn_test = [-1.297, -1.299, -1.334, -24.171]
        if name == 'HSE03':
            de_skn_test = [-2.390, -2.371, -2.420, -24.391]
        if name == 'HSE06':
            de_skn_test = [-1.963, -1.953, -1.997, -24.311]

        if rank == 0:
            print de_skn[0, 0, 1:4]
            print de_skn[0, 1, 2:4]
            print de_skn[0, 2, 2:4]
            print hyb_calc.exx
            assert abs(de_skn[0, 0, 1:4] - de_skn_test[0]).max() < 0.003
            assert abs(de_skn[0, 1, 2:4] - de_skn_test[1]).max() < 0.003
            assert abs(de_skn[0, 2, 2:4] - de_skn_test[2]).max() < 0.003
            assert abs(hyb_calc.exx - de_skn_test[3]) < 0.001
