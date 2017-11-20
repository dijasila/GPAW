import numpy as np

from gpaw.mixer import DummyMixer
from gpaw.xc import XC
from gpaw.xc.kernel import XCNull


class KickHamiltonian(object):
    def __init__(self, calc, ext):
        ham = calc.hamiltonian
        dens = calc.density
        vext_g = ext.get_potential(ham.finegd)
        self.vt_sG = [ham.restrict_and_collect(vext_g)]
        self.dH_asp = ham.setups.empty_atomic_matrix(1, ham.atom_partition)

        W_aL = dens.ghat.dict()
        dens.ghat.integrate(vext_g, W_aL)
        # XXX this is a quick hack to get the distribution right
        dHtmp_asp = ham.atomdist.to_aux(self.dH_asp)
        for a, W_L in W_aL.items():
            setup = dens.setups[a]
            dHtmp_asp[a] = np.dot(setup.Delta_pL, W_L).reshape((1, -1))
        self.dH_asp = ham.atomdist.from_aux(dHtmp_asp)


class TimeDependentHamiltonian(object):
    def __init__(self, fxc=None):
        self.fxc_name = fxc

    def write(self, writer):
        if self.has_fxc:
            self.write_fxc(writer.child('fxc'))

    def write_fxc(self, writer):
        wfs = self.wfs
        writer.write(name=self.fxc_name)
        M = wfs.setups.nao
        writer.add_array('deltaXC_H_MM',
                         (wfs.nspins, wfs.kd.nibzkpts, M, M),
                         dtype=wfs.dtype)
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                H_MM = wfs.collect_auxiliary('deltaXC_H_MM', k, s,
                                             shape=(M, M),
                                             dtype=wfs.dtype)
                writer.fill(H_MM)

    def read(self, reader):
        if 'fxc' in reader:
            self.read_fxc(reader.fxc)

    def read_fxc(self, reader):
        assert self.fxc_name is None or self.fxc_name == reader.name
        self.fxc_name = reader.name
        wfs = self.wfs
        for kpt in wfs.kpt_u:
            kpt.deltaXC_H_MM = reader.proxy('deltaXC_H_MM', kpt.s, kpt.k)[:]

    def initialize(self, paw):
        self.timer = paw.timer
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.hamiltonian
        niter = paw.niter

        # Reset the density mixer
        self.density.set_mixer(DummyMixer())
        self.update()

        # Initialize fxc
        self.initialize_fxc(niter)

    def initialize_fxc(self, niter):
        self.has_fxc = self.fxc_name is not None
        if not self.has_fxc:
            return

        get_H_MM = self.get_hamiltonian_matrix

        # Calculate deltaXC: 1. take current H_MM
        if niter == 0:
            for k, kpt in enumerate(self.wfs.kpt_u):
                kpt.deltaXC_H_MM = get_H_MM(kpt, addfxc=False)

        # Update hamiltonian.xc
        if self.fxc_name == 'RPA':
            xc = XCNull()
        else:
            xc = self.fxc_name
        self.hamiltonian.xc = XC(xc)
        self.update()

        # Calculate deltaXC: 2. update with new H_MM
        if niter == 0:
            for k, kpt in enumerate(self.wfs.kpt_u):
                kpt.deltaXC_H_MM -= get_H_MM(kpt, addfxc=False)

    def update_projectors(self):
        self.timer.start('LCAO update projectors')
        # Loop over all k-points
        for k, kpt in enumerate(self.wfs.kpt_u):
            self.wfs.atomic_correction.calculate_projections(self.wfs, kpt)
        self.timer.stop('LCAO update projectors')

    def get_hamiltonian_matrix(self, kpt, addfxc=True):
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        H_MM = get_matrix(self.hamiltonian, self.wfs, kpt, root=-1)
        if addfxc and self.has_fxc:
            H_MM += kpt.deltaXC_H_MM
        return H_MM

    def update(self):
        self.update_projectors()
        self.density.update(self.wfs)
        self.hamiltonian.update(self.density)
