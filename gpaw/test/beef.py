from __future__ import division
import numpy as np
from ase.build import bulk
from ase.dft.bee import BEEFEnsemble, readbee
from ase.eos import EquationOfState as EOS
from gpaw import GPAW
from gpaw.test import gen
from gpaw.mpi import world
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('MGGA_X_MBEEF') is not None

gen('Si', xcname='PBEsol')

results = {'mBEEF': (5.444, 0.052),
           'BEEF-vdW': (5.482, 0.072),
           'mBEEF-vdW': (5.426, 0.025)}

for xc in ['mBEEF', 'BEEF-vdW', 'mBEEF-vdW']:
    print(xc)
    if not newlibxc and xc[0] == 'm':
        print('Skipped', xc)
        continue

    if xc == 'mBEEF-vdW':
        # Does not work with libxc-4
        continue

    E = []
    V = []
    for a in np.linspace(5.4, 5.5, 5):
        si = bulk('Si', a=a)
        si.calc = GPAW(txt='Si-' + xc + '.txt',
                       xc=xc,
                       kpts=[2, 2, 2],
                       mode='pw')
        E.append(si.get_potential_energy())
        ens = BEEFEnsemble(si.calc, verbose=False)
        ens.get_ensemble_energies()
        ens.write('Si-{}-{:.3f}'.format(xc, a))
        V.append(si.get_volume())

    eos = EOS(V, E)
    v0, e0, B = eos.fit()
    a = (v0 * 4)**(1 / 3)

    a0, da0 = results[xc]

    assert abs(a - a0) < 0.001, (xc, a, a0)

    if world.rank == 0:
        E = []
        for a in np.linspace(5.4, 5.5, 5):
            e = readbee('Si-{}-{:.3f}'.format(xc, a))
            E.append(e)

        A = []
        for energies in np.array(E).T:
            eos = EOS(V, energies, 'sj')
            try:
                v0, e0, B = eos.fit()
                A.append((v0 * 4)**(1 / 3))
            except ValueError:
                pass
        A = np.array(A)
        a = A.mean()
        da = A.std()
        print('a(ref) = {:.3f} +- {:.3f}'.format(a0, da0))
        print('a      = {:.3f} +- {:.3f}'.format(a, da))
        assert abs(a - a0) < 0.01
        assert abs(da - da0) < 0.01
