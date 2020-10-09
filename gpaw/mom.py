import warnings

import numpy as np

from ase.units import Ha

from gpaw.occupations import FixedOccupationNumbers


def mom_calculation(calc,
                    atoms,
                    occupations,
                    constraints=None,
                    space='full',
                    width=0.0,
                    width_increment=0.01,
                    niter_smearing=None):

    if space == 'reduced':
        assert constraints is not None, \
            'Provide constraints as MOMConstraint objects'
    if space == 'full' and width != 0.0:
        warnings.warn("Smearing not available for space='full'")

    if calc.wfs is None:
        calc.initialize(atoms)

    parallel_layout = calc.wfs.occupations.parallel_layout
    occ = FixedOccupationNumbers(occupations, parallel_layout)
    occ_mom = OccupationsMOM(calc, occ,
                             occupations,
                             constraints,
                             space, width,
                             width_increment,
                             niter_smearing)
    if calc.scf.converged:
        calc.set(occupations=occ_mom)
    else:
        calc.wfs.occupations = occ_mom

    calc.log(occ_mom)


class OccupationsMOM:
    def __init__(self,
                 calc,
                 occ,
                 occupations,
                 constraints=None,
                 space='full',
                 width=0.0,
                 width_increment=0.01,
                 niter_smearing=None):
        self.wfs = calc.wfs
        self.occ = occ
        self.extrapolate_factor = occ.extrapolate_factor
        self.occupations = np.array(occupations)
        self.constraints = constraints
        self.space = space
        self.width = width / Ha
        self.width_increment = width_increment / Ha
        self.niter_smearing = niter_smearing

        self.name = 'mom'
        self.iters = 0
        self.ne = None

    def todict(self):
        dct = {'name': self.name}
        if self.width != 0.0:
            dct['width'] = self.width * Ha
        return dct

    def __str__(self):
        s = 'Excited-state calculation with Maximum Overlap Method\n'
        s += '  Smearing of constraints: '
        if self.width == 0.0:
            s += 'off\n'
        else:
            s += '{0:.4f} eV\n'.format(self.width * Ha)
        return s

    def calculate(self,
                  nelectrons,
                  eigenvalues,
                  weights,
                  fermi_levels_guess):
        f_sn = self.occupations.copy()

        if self.iters == 0 and self.space == 'full':
            self.f_sn_unique = {}
            for s, f_n in enumerate(f_sn):
                self.f_sn_unique[s] = {}
                for f_n_unique in np.unique(f_n):
                    if f_n_unique >= 1.0e-10:
                        self.f_sn_unique[s][f_n_unique] = f_n == f_n_unique

            self.initialize_reference_orbitals()

        for kpt in self.wfs.kpt_u:
            if self.space == 'full':
                if self.iters == 0:
                    continue
                else:
                    f_sn[kpt.s].fill(0)

                    # Compute projections within equally occupied subspaces
                    for f_n_unique in self.f_sn_unique[kpt.s]:
                        occupied = self.f_sn_unique[kpt.s][f_n_unique]
                        n_occ = len(f_sn[kpt.s][occupied])
                        P = self.calculate_mom_projections(kpt, f_n_unique)
                        P_max = np.argpartition(P, -n_occ)[-n_occ:]
                        P_max.sort() # Do we need this?

                        f_sn[kpt.s][P_max] = f_n_unique

            elif self.space == 'reduced':
                for c in self.constraints:
                    if c[2] != kpt.s:
                        continue

                    orb = c[1]
                    max_overlap = orb.get_maximum_overlap(self.wfs, kpt,
                                                          c[0], self.iters)

                    occ_new = f_sn[kpt.s][max_overlap] + c[0]
                    if (occ_new < 0.0) or (occ_new > kpt.weight):
                        continue

                    if self.width != 0.0:
                        # Gaussian smearing of constraints
                        mask, gauss = self.smear_gaussian(kpt, f_sn,
                                                          c[0], max_overlap)
                        f_sn[kpt.s][mask] += (c[0] * gauss)
                    else:
                        f_sn[kpt.s][max_overlap] += c[0]

        if self.ne is None:
            self.ne = f_sn.sum(1)
        else:
            # TODO: Works only for spin polarized calculations
            for kpt in self.wfs.kpt_u:
                 f_sn[kpt.s] = self.check_number_of_electrons(kpt,
                                                              f_sn[kpt.s])

        self.occ.f_sn = f_sn
        f_qn, fermi_levels, e_entropy = self.occ.calculate(nelectrons,
                                                           eigenvalues,
                                                           weights,
                                                           fermi_levels_guess)

        self.iters += 1

        return f_qn, fermi_levels, e_entropy

    def initialize_reference_orbitals(self):
        if self.wfs.mode == 'lcao':
            self.c_ref = {}

            for kpt in self.wfs.kpt_u:
                self.c_ref[kpt.s] = {}

                # Initialize ref orbitals for each equally occupied subspace
                for f_n_unique in self.f_sn_unique[kpt.s]:
                    occupied = self.f_sn_unique[kpt.s][f_n_unique]
                    self.c_ref[kpt.s][f_n_unique] = kpt.C_nM[occupied].copy()
        else:
            self.wf = {}
            self.p_an = {}
            for kpt in self.wfs.kpt_u:
                self.wf[kpt.s] = {}
                self.p_an[kpt.s] = {}

                # Initialize ref orbitals for each equally occupied subspace
                for f_n_unique in self.f_sn_unique[kpt.s]:
                    occupied = self.f_sn_unique[kpt.s][f_n_unique]
                    # Pseudo wave functions
                    self.wf[kpt.s][f_n_unique] = kpt.psit_nG[occupied].copy()
                    # PAW projection
                    self.p_an[kpt.s][f_n_unique] = \
                        dict([(a, np.dot(self.wfs.setups[a].dO_ii,
                                         P_ni[occupied].T))
                              for a, P_ni in kpt.P_ani.items()])

    def calculate_mom_projections(self, kpt, f_n_unique):
        if self.wfs.mode == 'lcao':
            P = np.dot(self.c_ref[kpt.s][f_n_unique].conj(),
                       np.dot(kpt.S_MM, kpt.C_nM.T))
            P = np.sum(P**2, axis=0)
            P = P ** 0.5
        else:
            # Pseudo wave functions overlaps
            P = self.wfs.integrate(self.wf[kpt.s][f_n_unique][:],
                                   kpt.psit_nG[:], True)

            # PAW corrections
            P_corr = np.zeros_like(P)
            for a, p_a in self.p_an[kpt.s][f_n_unique].items():
                P_corr += np.dot(kpt.P_ani[a].conj(), p_a).T
            P_corr = np.ascontiguousarray(P_corr)
            self.wfs.gd.comm.sum(P_corr)

            # Sum pseudo wave and PAW contributions
            P += P_corr
            P = np.sum(P ** 2, axis=0)
            P = P ** 0.5

        return P

    def sort_wavefunctions(self, wfs, kpt):
        occupied = kpt.f_n > 1.0e-10
        n_occ = len(kpt.f_n[occupied])

        if n_occ == 0.0:
            return

        if np.min(kpt.f_n[:n_occ]) == 0:
            ind_occ = np.argwhere(occupied)
            ind_unocc = np.argwhere(~occupied)
            ind = np.vstack((ind_occ, ind_unocc))

            # Sort coefficients, occupation numbers, eigenvalues
            if wfs.mode == 'lcao':
                kpt.C_nM = np.squeeze(kpt.C_nM[ind])
            else:
                kpt.psit_nG[:] = np.squeeze(kpt.psit_nG[ind])
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            kpt.f_n = np.squeeze(kpt.f_n[ind])
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
            self.c_n = kpt.C_nM[self.n].copy()
        else:
            self.wf_n = kpt.psit_nG[self.n].copy()
            self.p_an = dict([(a, np.dot(wfs.setups[a].dO_ii, P_ni[self.n]))
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
                P_n[ini:fin] = np.dot(self.c_n.conj(),
                                  np.dot(kpt.S_MM, kpt.C_nM[ini:fin].T))
        else:
            # Pseudo wave functions overlaps
            P_n[ini:fin] = wfs.integrate(self.wf_n, kpt.psit_nG[ini:fin], False)

            if iters > 1:
                # Add PAW corrections
                for a, p_a in self.p_an.items():
                    P_n[ini:fin] += np.dot(kpt.P_ani[a][ini:fin].conj(), p_a)
            wfs.gd.comm.sum(P_n)

        P_n = P_n ** 2

        # Update index of target orbital
        self.n = np.argmax(P_n)

        if iters == 1:
            # If positions have changed than overlap operators change
            # So reinitialize reference orbitals
            self.update_target_orbital(wfs, kpt)

        return self.n

