import numpy as np

from gpaw.mixer import DummyMixer
from gpaw.xc import XC

from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import write_uMM


class TimeDependentPotential(object):
    def __init__(self, ext, laser):
        self.ext = ext
        self.laser = laser

    def initialize(self, paw):
        get_matrix = paw.wfs.eigensolver.calculate_hamiltonian_matrix
        hamiltonian = KickHamiltonian(paw.hamiltonian, paw.density, self.ext)
        self.V_uMM = []
        for kpt in paw.wfs.kpt_u:
            V_MM = get_matrix(hamiltonian, paw.wfs, kpt,
                              add_kinetic=False, root=-1)
            self.V_uMM.append(V_MM)

    def get_MM(self, u, time):
        return self.laser.strength(time) * self.V_uMM[u]

    def write(self, writer):
        writer.write(info='this is not implemented')


class KickHamiltonian(object):
    def __init__(self, ham, dens, ext):
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
    def __init__(self, fxc=None, td_potential=None):
        assert fxc is None or isinstance(fxc, str)
        self.fxc_name = fxc
        if isinstance(td_potential, dict):
            td_potential = TimeDependentPotential(**td_potential)
        self.td_potential = td_potential

    def write(self, writer):
        if self.has_fxc:
            self.write_fxc(writer.child('fxc'))
        if self.td_potential is not None:
            self.td_potential.write(writer.child('td_potential'))

    def write_fxc(self, writer):
        wfs = self.wfs
        writer.write(name=self.fxc_name)
        write_uMM(wfs, writer, 'deltaXC_H_uMM', self.deltaXC_H_uMM)

    def read(self, reader):
        if 'fxc' in reader:
            self.read_fxc(reader.fxc)
        if 'td_potential' in reader:
            raise NotImplementedError('Restart with td_potential is not '
                                      'implemented.')

    def read_fxc(self, reader):
        assert self.fxc_name is None or self.fxc_name == reader.name
        self.fxc_name = reader.name
        self.deltaXC_H_uMM = read_uMM(self.wfs, reader, 'deltaXC_H_uMM')

    def initialize(self, paw):
        self.timer = paw.timer
        self.timer.start('Initialize TDDFT Hamiltonian')
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.hamiltonian
        self.occupations = paw.occupations
        niter = paw.niter

        # Reset the density mixer
        # XXX: density mixer is not written to the gpw file
        # XXX: so we need to set it always
        self.density.set_mixer(DummyMixer())
        self.update()

        # Initialize fxc
        self.initialize_fxc(niter)

        # Initialize td_potential
        if self.td_potential is not None:
            self.td_potential.initialize(paw)

        self.timer.stop('Initialize TDDFT Hamiltonian')

    def initialize_fxc(self, niter):
        self.has_fxc = self.fxc_name is not None
        if not self.has_fxc:
            return
        self.timer.start('Initialize fxc')
        # XXX: Similar functionality is available in
        # paw.py: PAW.linearize_to_xc(self, newxc)
        # See test/lcaotddft/fxc_vs_linearize.py

        # Calculate deltaXC: 1. take current H_MM
        if niter == 0:
            def get_H_MM(kpt):
                return self.get_hamiltonian_matrix(kpt, time=0.0,
                                                   addfxc=False, addpot=False)
            self.deltaXC_H_uMM = []
            for kpt in self.wfs.kpt_u:
                self.deltaXC_H_uMM.append(get_H_MM(kpt))

        # Update hamiltonian.xc
        if self.fxc_name == 'RPA':
            xc_name = 'null'
        else:
            xc_name = self.fxc_name
        # XXX: xc is not written to the gpw file
        # XXX: so we need to set it always
        xc = XC(xc_name)
        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)
        xc.set_positions(self.hamiltonian.spos_ac)
        self.hamiltonian.xc = xc
        self.update()

        # Calculate deltaXC: 2. update with new H_MM
        if niter == 0:
            for u, kpt in enumerate(self.wfs.kpt_u):
                self.deltaXC_H_uMM[u] -= get_H_MM(kpt)
        self.timer.stop('Initialize fxc')

    def update_projectors(self):
        self.timer.start('Update projectors')
        for kpt in self.wfs.kpt_u:
            self.wfs.atomic_correction.calculate_projections(self.wfs, kpt)
        self.timer.stop('Update projectors')

    def get_hamiltonian_matrix(self, kpt, time, addfxc=True, addpot=True):
        self.timer.start('Calculate H_MM')
        kpt_rank, u = self.wfs.kd.get_rank_and_index(kpt.s, kpt.k)
        assert kpt_rank == self.wfs.kd.comm.rank

        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        H_MM = get_matrix(self.hamiltonian, self.wfs, kpt, root=-1)
        if addfxc and self.has_fxc:
            H_MM += self.deltaXC_H_uMM[u]
        if addpot and self.td_potential is not None:
            H_MM += self.td_potential.get_MM(u, time)
        self.timer.stop('Calculate H_MM')
        return H_MM

    def update(self, mode='all'):
        self.timer.start('Update TDDFT Hamiltonian')
        if mode in ['all', 'density']:
            self.update_projectors()
            self.density.update(self.wfs)
        if mode in ['all']:
            self.hamiltonian.update(self.density)
        self.timer.stop('Update TDDFT Hamiltonian')
