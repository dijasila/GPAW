import numpy as np
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, PW
from gpaw.mpi import size, rank, world, serial_comm
from gpaw.xc.tools import vxc
from gpaw.xc.hyb import HybridXC


mgo = bulk('MgO', 'rocksalt', a=4.189)
comm = world.new_communicator(np.arange(min(3, size)))
if rank < 3:
    if 1:
        mgo.calc = GPAW(mode=PW(500),
                        parallel=dict(band=1),
                        idiotproof=False,
                        communicator=comm,
                        kpts=monkhorst_pack((2, 2, 2)) + 0.25)
        mgo.get_potential_energy()
        mgo.calc.write('mgo', 'all')

    calc = GPAW('mgo', txt=None, communicator=serial_comm)
    pbe0 = HybridXC('PBE0', alpha=5.0, bandstructure=True, world=comm)
    de_skn = vxc(calc, pbe0) - vxc(calc, 'LDA')
    if rank == 0:
        assert abs(de_skn[0, 0, 1:4] - -1.348).max() < 0.004
        assert abs(de_skn[0, 1, 2:4] - -1.350).max() < 0.003
        assert abs(de_skn[0, 2, 2:4] - -1.367).max() < 0.0005
        assert abs(pbe0.exx - -24.18633) < 2e-5
