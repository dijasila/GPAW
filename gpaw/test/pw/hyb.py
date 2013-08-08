from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import rank, size, serial_comm
from gpaw.xc.hybridg import HybridXC
from gpaw.test import equal

a = 2.0
li = Atoms('Li', cell=(a, a, a), pbc=1)
for spinpol in [False, True]:
    for usesymm in [True, False, None]:
        if size == 8 and not spinpol and usesymm:
            continue
        for qparallel in [False, True]:
            if rank == 0:
                print(spinpol, usesymm, qparallel)
            li.calc = GPAW(mode=PW(300),
                           kpts=(2, 3, 4),
                           spinpol=spinpol,
                           usesymm=usesymm,
                           parallel={'band': 1},
                           txt=None,
                           idiotproof=False)
            e = li.get_potential_energy()
            if qparallel:
                li.calc.write('li', mode='all')
                calc = GPAW('li', txt=None, communicator=serial_comm)
            else:
                calc = li.calc
            exx = HybridXC('EXX',
                           logfilename=None,
                           method='acdf')
            de = calc.get_xc_difference(exx)
            exx = HybridXC('EXX',
                           logfilename=None,
                           method='acdf',
                           bandstructure=True, bands=[0, 1])
            de2 = calc.get_xc_difference(exx)
            kd = calc.wfs.kd
            equal(e, -0.52, 0.05)
            equal(de, -0.44, 0.02)
            equal(de, de2, 1e-12)
            for k in range(kd.nibzkpts):
                if abs(kd.ibzk_kc[k] - [0.25, 1 / 3.0, 3 / 8.0]).max() < 1e-7:
                    equal(abs(exx.exx_skn[:, k, 0] - -0.1826).max(), 0, 1e-4)
