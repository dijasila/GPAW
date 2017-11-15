from ase.utils.timing import timer

from gpaw.analyse.observers import Observer


class DipoleMomentWriter(Observer):

    def __init__(self, filename, paw):
        Observer.__init__(self)
        assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT'
        self.timer = paw.timer
        self.master = paw.world.rank == 0
        if self.master:
            if paw.niter == 0:
                self.fd = open(filename, 'w')
            else:
                self.fd = open(filename, 'a')

        paw.attach(self, 1, paw)

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

    def _write_dm(self, paw):
        time = paw.time
        density = paw.density
        norm = density.finegd.integrate(density.rhot_g)
        dm = density.finegd.calculate_dipole_moment(density.rhot_g)
        line = ('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                (time, norm, dm[0], dm[1], dm[2]))
        self._write(line)

    @timer('DipoleMomentWriter update')
    def update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_dm(paw)

    def close(self):
        if self.master:
            self.fd.close()

    def __del__(self):
        self.close()
