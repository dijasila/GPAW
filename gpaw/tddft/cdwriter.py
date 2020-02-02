import re
import numpy as np

from ase.units import Bohr
from gpaw.utilities.tools import coordinates
from gpaw.fd_operators import Gradient
from gpaw.lcaotddft.observer import TDDFTObserver


def convert_repr(r):
    # Integer
    try:
        return int(r)
    except ValueError:
        pass
    # Boolean
    b = {repr(False): False, repr(True): True}.get(r, None)
    if b is not None:
        return b
    # String
    s = r[1:-1]
    if repr(s) == r:
        return s
    raise RuntimeError('Unknown value: %s' % r)


def skew(a):
    return (a - a.T) / 2


class CDWriter(TDDFTObserver):
    version = 1

    def __init__(self, paw, filename, center=True,
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
        if paw.niter == 0:
            # Initialize
            self.do_center = center
            if self.master:
                self.fd = open(filename, 'w')
        else:
            # Read and continue
            self.read_header(filename)
            if self.master:
                self.fd = open(filename, 'a')

        gd = paw.wfs.gd

        assert center
        R0 = 0.5 * np.diag(gd.cell_cv)
        r_cG, _ = coordinates(gd, origin=R0)
        self.r_cG = r_cG
        self.Ra_a = paw.atoms.positions / Bohr - R0[None, :]

        # Create Gradient operator
        gd = paw.wfs.gd
        grad_v = []
        for v in range(3):
            grad_v.append(Gradient(gd, v, dtype=complex, n=2))
        self.grad_v = grad_v
        self.timer = paw.timer

    def _write(self, line):
        if self.master:
            self.fd.write(line)
            self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = '# %s[version=%s]' % (self.__class__.__name__, self.version)
        line += ('(center=%s)\n' %
                 (repr(self.do_center)))
        line += ('# %15s %22s %22s %22s\n' %
                 ('time', 'cmx', 'cmy', 'cmz'))
        self._write(line)

    def read_header(self, filename):
        with open(filename, 'r') as f:
            line = f.readline()
        m_i = re.split("[^a-zA-Z0-9_=']+", line[2:])
        assert m_i.pop(0) == self.__class__.__name__
        for m in m_i:
            if '=' not in m:
                continue
            k, v = m.split('=')
            v = convert_repr(v)
            if k == 'version':
                assert v == self.version
                continue
            # Translate key
            k = {'center': 'do_center'}[k]
            setattr(self, k, v)

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def calculate_cd_moment(self, paw):
        grad_v = self.grad_v
        wfs = paw.wfs
        gd = wfs.gd
        bd = wfs.bd
        r_cG = self.r_cG
        Ra_a = self.Ra_a

        self.timer.start('CD')

        grad_psit_vG = gd.empty(3, dtype=complex)
        rxnabla_a = np.zeros(3, dtype=complex)
        # Ra x <psi1| nabla |psi2>
        Rxnabla_a = np.zeros(3, dtype=complex)

        rxnabla_g = np.zeros(3, dtype=complex)

        for kpt in wfs.kpt_u:
            for n, (f, psit_G) in enumerate(zip(kpt.f_n, kpt.psit_nG)):
                pref = -1j * f

                for v in range(3):
                    grad_v[v].apply(psit_G, grad_psit_vG[v], kpt.phase_cd)

                self.timer.start('Pseudo')

                def calculate(v1, v2):
                    return pref * gd.integrate(psit_G.conjugate() *
                                               (r_cG[v1] * grad_psit_vG[v2] -
                                                r_cG[v2] * grad_psit_vG[v1]))

                rxnabla_g[0] += calculate(1, 2)
                rxnabla_g[1] += calculate(2, 0)
                rxnabla_g[2] += calculate(0, 1)
                self.timer.stop('Pseudo')

                # augmentation contributions to magnetic moment
                # <psi1| r x nabla |psi2> = <psi1| (r - Ra + Ra) x nabla |psi2>
                #                         = <psi1| (r - Ra) x nabla |psi2>
                #                             + Ra x <psi1| nabla |psi2>

                self.timer.start('PAW')
                # <psi1| (r-Ra) x nabla |psi2>
                for a, P_ni in kpt.P_ani.items():
                    Ra = Ra_a[a]
                    P_i = P_ni[n]

                    rxnabla_iiv = wfs.setups[a].rxnabla_iiv.copy()
                    nabla_iiv = wfs.setups[a].nabla_iiv.copy()
                    for v in range(3):
                        rxnabla_iiv[:, :, v] = skew(rxnabla_iiv[:, :, v])
                        nabla_iiv[:, :, v] = skew(nabla_iiv[:, :, v])

                    for i1, P1 in enumerate(P_i):
                        for i2, P2 in enumerate(P_i):
                            PP = P1.conjugate() * P2
                            rxnabla_v = rxnabla_iiv[i1, i2]
                            nabla_v = nabla_iiv[i1, i2]

                            for v in range(3):
                                rxnabla_a[v] += pref * PP * rxnabla_v[v]

                            def calculate(v1, v2):
                                return pref * PP * (Ra[v1] * nabla_v[v2] -
                                                    Ra[v2] * nabla_v[v1])

                            # (y pz - z py)i + (z px - x pz)j + (x py - y px)k
                            Rxnabla_a[0] += calculate(1, 2)
                            Rxnabla_a[1] += calculate(2, 0)
                            Rxnabla_a[2] += calculate(0, 1)
                self.timer.stop('PAW')

        bd.comm.sum(rxnabla_a)
        gd.comm.sum(rxnabla_a)

        bd.comm.sum(Rxnabla_a)
        gd.comm.sum(Rxnabla_a)

        bd.comm.sum(rxnabla_g)

        rxnabla_tot = rxnabla_g + Rxnabla_a + rxnabla_a

        self.timer.stop('CD')

        return rxnabla_tot.real

    def _write_cd(self, paw):
        time = paw.time

        cd = self.calculate_cd_moment(paw)
        line = ('%20.8lf %22.12le %22.12le %22.12le\n' %
                (time, cd[0], cd[1], cd[2]))
        self._write(line)

    def _update(self, paw):
        # if paw.action == 'init':
        #     self._write_header(paw)
        # elif paw.action == 'kick':
        #     self._write_kick(paw)
        self._write_cd(paw)

    def __del__(self):
        if self.master:
            self.fd.close()
        TDDFTObserver.__del__(self)
