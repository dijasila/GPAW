import re
import numpy as np

from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.dipolemomentwriter import convert_repr


def calculate_quadrupole_moment(gd, rho_g, center_v):
    # Center relative to the cell center
    # center_v += 0.5 * gd.cell_cv.sum(0)
    r_vg = gd.get_grid_point_coordinates()
    qm_i = np.zeros(6, dtype=float)
    i = 0
    for v1 in range(3):
        x1_g = r_vg[v1] - center_v[v1]
        for v2 in range(v1, 3):
            x2_g = r_vg[v2] - center_v[v2]
            qm_i[i] = -gd.integrate(x1_g * x2_g * rho_g)
            i += 1
    return qm_i



class QuadrupoleMomentWriter(TDDFTObserver):
    version = 1

    def __init__(self, paw, filename, center=[0, 0, 0], density='comp',
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
        if paw.niter == 0:
            # Initialize
            self.center_v = center
            self.density_type = density
            if self.master:
                self.fd = open(filename, 'w')
        else:
            # Read and continue
            self.read_header(filename)
            if self.master:
                self.fd = open(filename, 'a')

    def _write(self, line):
        if self.master:
            self.fd.write(line)
            self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = '# %s[version=%s]' % (self.__class__.__name__, self.version)
        line += ('(center=%s, density=%s)\n' %
                 (repr(self.center_v), repr(self.density_type)))
        line += ('# %15s %15s %22s %22s %22s %22s %22s %22s\n' %
                 ('time', 'norm', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2'))
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
            k = {'center': 'center_v', 'density': 'density_type'}[k]
            setattr(self, k, v)

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def _write_dm(self, paw):
        time = paw.time
        density = paw.density
        if self.density_type == 'comp':
            rho_g = density.rhot_g
            gd = density.finegd
        elif self.density_type == 'pseudo':
            rho_g = density.nt_sg.sum(axis=0)
            gd = density.finegd
        elif self.density_type == 'pseudocoarse':
            rho_g = density.nt_sG.sum(axis=0)
            gd = density.gd
        else:
            raise RuntimeError('Unknown density type: %s' % self.density_type)

        norm = gd.integrate(rho_g)
        qm = calculate_quadrupole_moment(gd, rho_g, center_v=self.center_v)
        line = (('%20.8lf %20.8le' + ' %22.12le' * len(qm) + '\n') %
                ((time, norm) + tuple(qm)))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_dm(paw)

    def __del__(self):
        if self.master:
            self.fd.close()
        TDDFTObserver.__del__(self)
