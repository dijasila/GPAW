import numpy as np
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack

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
                        setups={'Mg': '2'},
                        xc = 'PBE', 
                        convergence={'eigenstates': 5.e-9},
                        kpts=monkhorst_pack((2, 2, 2)) + 0.25)
        mgo.get_potential_energy()
        mgo.calc.write('mgo', 'all')

    calc = GPAW('mgo', txt=None, communicator=serial_comm)
    hse06 = HybridXC('HSE06', bandstructure=True, world=comm)
    de_skn = vxc(calc, hse06) - vxc(calc, 'PBE')
    if rank == 0:
        print de_skn[0, 0, 1:4], -2.1465
        print de_skn[0, 1, 2:4], -2.2000
        print de_skn[0, 2, 2:4], -2.1745
        print hse06.exx, -24.414829
        print de_skn
        assert abs(de_skn[0, 0, 1:4] - -2.1465).max() < 0.02
        assert abs(de_skn[0, 1, 2:4] - -2.2000).max() < 0.008
        assert abs(de_skn[0, 2, 2:4] - -2.1745).max() < 0.004
        assert abs(hse06.exx - -24.414829) < 2e-4
