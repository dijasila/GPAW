import warnings
import numpy as np
from ase.units import Hartree
from gpaw.occupations import ZeroKelvin


class OccupationsMOM(ZeroKelvin):
    def __init__(self, occupations,
                 constraints=None,
                 space='reduced',
                 width=0.0,
                 width_increment=0.01,
                 niter_smearing=None):
        ZeroKelvin.__init__(self, True)
        self.occupations = np.array(occupations)
        self.constraints = constraints
        self.space = space
        self.width = width / Hartree
        self.width_increment = width_increment / Hartree
        self.niter_smearing = niter_smearing

        self.name = 'mom'
        self.iters = 0
        self.ne = None

        if self.space == 'reduced':
            assert self.constraints is not None, \
                'Provide constraints as MOMConstraint objects'
        if self.space == 'full' and self.width != 0:
            warnings.warn("Smearing is not used when space='full'")

    def todict(self):
        dct = {'name': self.name}
        if self.width != 0.0:
            dct['width'] = self.width * Hartree
        return dct

    def __str__(self):
        s = 'Occupation numbers:\n'
        s += '  Delta SCF with Maximum Overlap Method\n'
        s += '  Smearing of constraints: '
        if self.width == 0.0:
            s += 'off\n'
        else:
            s += '{0:.4f} eV\n'.format(self.width * Hartree)
        return s

    def calculate(self, wfs):
        occ = self.occupations.copy()

        if self.iters == 1 and self.space == 'full':
            self.initialize_reference_orbitals(wfs)

        for kpt in wfs.kpt_u:
            if self.space == 'full':
                if self.iters == 0:
                    continue
                else:
                    occ[kpt.s].fill(0)

                    # Compute projections within each occupied subspace
                    for f in self.c_ref[kpt.s].keys():
                        n_occ = len(self.c_ref[kpt.s][f])
                        P = self.calculate_projections(wfs, kpt, f)
                        P_max = np.argpartition(P, -n_occ)[-n_occ:]
                        P_max.sort() # Do we need this?

                        occ[kpt.s][P_max] = f

            elif self.space == 'reduced':
                for c in self.constraints:
                    if c[2] != kpt.s:
                        continue

                    orb = c[1]
                    max_overlap = orb.get_maximum_overlap(wfs, kpt,
                                                          c[0], self.iters)

                    occ_new = occ[kpt.s][max_overlap] + c[0]
                    if (occ_new < 0.0) or (occ_new > kpt.weight):
                        continue

                    if self.width != 0.0:
                        # Gaussian smearing of constraints
                        mask, gauss = self.smear_gaussian(kpt, occ,
                                                          c[0], max_overlap)
                        occ[kpt.s][mask] += (c[0] * gauss)
                    else:
                        occ[kpt.s][max_overlap] += c[0]

        if self.ne is None:
            self.ne = occ.sum(1)
        else:
            # TODO: Works only for spin polarized calculations
            for kpt in wfs.kpt_u:
                 occ[kpt.s] = self.check_number_of_electrons(kpt, occ[kpt.s])

        self.occupation = occ
        ZeroKelvin.calculate(self, wfs)

        self.iters += 1

    def spin_paired(self, wfs):
        return self.fixed_moment(wfs)

    def fixed_moment(self, wfs):
        egs_name = getattr(wfs.eigensolver, "name", None)
        magmom = 0.0

        for kpt in wfs.kpt_u:
            wfs.bd.distribute(self.occupation[kpt.s], kpt.f_n)

            # Compute the magnetic moment
            if wfs.nspins == 2:
                if kpt.s == 0:
                    magmom += self.occupation[kpt.s].sum()
                else:
                    magmom -= self.occupation[kpt.s].sum()

        self.magmom = wfs.kptband_comm.sum(magmom)

    def initialize_reference_orbitals(self, wfs):
        self.c_ref = {}

        if wfs.mode == 'lcao':
            for kpt in wfs.kpt_u:
                self.c_ref[kpt.s] = {}

                # Initialize reference orbitals for each occupied subspace
                f_n_unique = np.unique(kpt.f_n)
                for f in f_n_unique:
                    if f >= 1.0e-10:
                        occupied = kpt.f_n == f

                        self.c_ref[kpt.s][f] = kpt.C_nM[occupied].copy()

    def calculate_projections(self, wfs, kpt, f):
        if wfs.mode == 'lcao':
            P = np.dot(self.c_ref[kpt.s][f].conj(),
                       np.dot(kpt.S_MM, kpt.C_nM.T))
            P = np.sum(P**2, axis=0)
            P = P ** 0.5
        else:
            raise NotImplementedError('Full MOM available only in LCAO mode')

        return P

    def sort_wavefunctions(self, kpt, sort_eps=False):
        occupied = kpt.f_n > 1.0e-10
        n_occ = len(kpt.f_n[occupied])

        if n_occ == 0.0:
            return

        if np.min(kpt.f_n[:n_occ]) == 0:
            ind_occ = np.argwhere(occupied)
            ind_unocc = np.argwhere(~occupied)
            ind = np.vstack((ind_occ, ind_unocc))

            # Sort coefficients, occupation numbers, eigenvalues
            kpt.C_nM = np.squeeze(kpt.C_nM[ind])
            kpt.f_n = np.squeeze(kpt.f_n[ind])
            if sort_eps:
                kpt.eps_n = np.squeeze(kpt.eps_n[ind])

    def smear_gaussian(self, kpt, occ, c, n):
        if c < 0:
            mask = (occ[kpt.s] != 0)
        else:
            mask = (occ[kpt.s] == 0)

        e = kpt.eps_n[mask]
        de2 = -(e - kpt.eps_n[n]) ** 2
        gauss = (1 / (self.width * np.sqrt(2 * np.pi)) *
                 np.exp(de2 / (2 * self.width ** 2)))
        gauss /= sum(gauss)

        return mask, gauss

    def check_number_of_electrons(self, kpt, occ):
        ne_diff = occ.sum() - self.ne[kpt.s]
        lumo = int(self.ne[kpt.s])
        homo = int(lumo - 1)

        # Check that total number of electrons is conserved
        while ne_diff != 0:
            if ne_diff < 0:
                occ[lumo] += 1
                lumo += 1
                ne_diff += 1
            else:
                occ[homo] -= 1
                homo -= 1
                ne_diff -= 1

        return occ

    def reset(self):
        if self.iters > 1:
            self.iters = 0

class MOMConstraint:
    def __init__(self, n, nstart=0, nend=None):
        self.n = n
        self.nstart = nstart
        self.nend = nend

    def initialize(self, wfs, c):
        nocc = wfs.nvalence // 2
        if self.nend is None:
            if c < 0:
                self.nend = nocc
            else:
                self.nend = self.nbands - nocc
        if c < 0:
            self.ini = self.nstart
            self.fin = self.nend
        else:
            self.ini = nocc + self.nstart
            self.fin = nocc + self.nend

    def update_target_orbital(self, wfs, kpt):
        if wfs.mode == 'lcao':
            self.c_u = kpt.C_nM[self.n].copy()
        else:
            self.wf_u = kpt.psit_nG[self.n].copy()
            self.p_uai = dict([(a, P_ni[self.n].copy())
                               for a, P_ni in kpt.P_ani.items()])

    def get_maximum_overlap(self, wfs, kpt, c, iters):
        self.nbands = wfs.bd.nbands

        if iters == 0:
            self.initialize(wfs, c)
            self.update_target_orbital(wfs, kpt)
        ini = self.ini
        fin = self.fin

        P_n = np.zeros(self.nbands)
        if wfs.mode == 'lcao':
            if kpt.S_MM is None:
                return self.n
            else:
                P_n[ini:fin] = np.dot(self.c_u.conj(),
                                  np.dot(kpt.S_MM, kpt.C_nM[ini:fin].T))
        else:
            # Pseudo wave functions overlaps
            wf = np.reshape(self.wf_u, -1)
            Wf_n = kpt.psit_nG
            Wf_n = np.reshape(Wf_n, (self.nbands, -1))
            P_n[ini:fin] = np.dot(Wf_n[ini:fin].conj(), wf) * wfs.gd.dv

            # Add PAW corrections
            for a, p_a in self.p_uai.items():
                p_ai = np.dot(wfs.setups[a].dO_ii, p_a)
                P_n[ini:fin] += np.dot(kpt.P_ani[a][ini:fin].conj(), p_ai)
        P_n = P_n ** 2

        # Update index of target orbital
        self.n = np.argmax(P_n)

        if iters == 1:
            # If positions have changed than overlap operators change
            # So reinitialize reference orbitals
            self.update_target_orbital(wfs, kpt)

        return self.n
