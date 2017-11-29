import numpy as np

from gpaw.lcaotddft.observer import TDDFTObserver


class DipoleMomentWriter(TDDFTObserver):

    def __init__(self, paw, filename, center=False, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
        self.do_center = center
        if self.master:
            if paw.niter == 0:
                self.fd = open(filename, 'w')
            else:
                self.fd = open(filename, 'a')

    def _write(self, line):
        if self.master:
            self.fd.write(line)
            self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = ('# %15s %15s %22s %22s %22s\n' %
                ('time', 'norm', 'dmx', 'dmy', 'dmz'))
        self._write(line)

    def _write_kick(self, paw):
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]\n' % tuple(kick)
        self._write(line)

    def calculate_dipole_moment(self, gd, rho_g):
        center_v = 0.5 * gd.cell_cv.sum(0)
        r_vg = gd.get_grid_point_coordinates()
        dm_v = np.zeros(3, dtype=float)
        for v in range(3):
            dm_v[v] = -gd.integrate((r_vg[v] - center_v[v]) * rho_g)
        return dm_v

    def _write_dm(self, paw):
        time = paw.time
        density = paw.density
        norm = density.finegd.integrate(density.rhot_g)
        if self.do_center:
            dm = self.calculate_dipole_moment(density.finegd, density.rhot_g)
        else:
            dm = density.finegd.calculate_dipole_moment(density.rhot_g)
        line = ('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                (time, norm, dm[0], dm[1], dm[2]))
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
