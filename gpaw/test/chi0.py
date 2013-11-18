import numpy as np
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack
from gpaw import GPAW
from gpaw.response.chi import CHI
from gpaw.response.chi0 import Chi0
from gpaw.mpi import serial_comm


omega = np.array([0, 1.0, 2.0])
for k in [2, 3]:
    q = [0, 0, 1.0 / k]
    for gamma in [False, True]:
        if k == 3 and gamma:
            continue
        kpts = monkhorst_pack((k, k, k))
        if gamma:
            kpts += 0.5 / k
        for center in [False, True]:
            a = bulk('Si', 'diamond')
            if center:
                a.center()
            for sym in [None, True]:
                name = 'si.k%d.g%d.c%d.s%d' % (k, gamma, center, bool(sym))
                print(name)
                if 0:
                    calc = a.calc = GPAW(kpts=kpts,
                                         usesymm=sym,
                                         mode='pw',
                                         width=0.001,
                                         txt=name + '.txt')
                    e = a.get_potential_energy()
                    #calc.diagonalize_full_hamiltonian(nbands=100)
                    calc.write(name, 'all')
                    
                calc = GPAW(name, txt=None, communicator=serial_comm)

                chi = CHI(calc, w=omega, q=q, ecut=100,
                          hilbert_trans=False, xc='RPA',
                          G_plus_q=True, txt=name + '.logold')
                chi.initialize()
                chi.calculate()
                chi0old_wGG = chi.chi0_wGG
                
                chi = Chi0(calc, omega, ecut=100, txt=name + '.log')
                pd, chi0_wGG, _ = chi.calculate(q)

                assert abs(chi0_wGG - chi0old_wGG).max() < 1e-15
                
                if sym is None and not center:
                    chi00_w = chi0_wGG[:, 0, 0]
                elif -1 not in calc.wfs.kd.bz2bz_ks:
                    assert abs(chi0_wGG[:, 0, 0] - chi00_w).max() < 3e-5
                    #print abs(chi0_wGG[:, 0, 0] - chi00_w).max()
                    
                if sym is None:
                    chi00_wGG = chi0_wGG
                elif -1 not in calc.wfs.kd.bz2bz_ks:
                    assert abs(chi0_wGG - chi00_wGG).max() < 2e-5
                    #print abs(chi0_wGG - chi00_wGG).max()
