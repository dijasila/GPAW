import numpy as np

from ase.units import Ha

from gpaw.occupations import FixedOccupationNumbers


def mom_calculation(calc, atoms,
                    numbers,
                    width=0.0,
                    width_increment=0.0,
                    niter_width_update=10):

    if calc.wfs is None:
        # We need the wfs object to initialize OccupationsMOM
        # so initialize calculator
        calc.initialize(atoms)

    parallel_layout = calc.wfs.occupations.parallel_layout
    occ = FixedOccupationNumbers(numbers, parallel_layout)

    occ_mom = OccupationsMOM(calc.wfs, occ,
                             numbers,
                             width,
                             width_increment,
                             niter_width_update)

    # Set MOM occupations and let calculator.py take care of the rest
    calc.set(occupations=occ_mom)

    calc.log(occ_mom)


class OccupationsMOM:
    def __init__(self, wfs, occ,
                 numbers,
                 width=0.0,
                 width_increment=0.0,
                 niter_width_update=10):
        self.wfs = wfs
        self.occ = occ
        self.extrapolate_factor = occ.extrapolate_factor
        self.numbers = np.array(numbers)
        self.width = width / Ha
        self.width_increment = width_increment / Ha
        self.niter_width_update = niter_width_update

        self.name = 'mom'
        self.iters = 0
        self.initialized = False

    def todict(self):
        dct = {'name': self.name,
               'numbers': self.numbers}
        if self.width != 0.0:
            dct['width'] = self.width * Ha
        return dct

    def __str__(self):
        s = 'Excited-state calculation with Maximum Overlap Method\n'
        s += '  Smearing of holes and excited electrons: '
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

        if not self.initialized:
            self.initialize_reference_orbitals()
            self.occ.f_sn = self.numbers.copy()
        else:
            self.occ.f_sn = self.update_occupations()
            self.iters += 1

        f_qn, fermi_levels, e_entropy = self.occ.calculate(nelectrons,
                                                           eigenvalues,
                                                           weights,
                                                           fermi_levels_guess)

        return f_qn, fermi_levels, e_entropy

    def initialize_reference_orbitals(self):
        if self.wfs.kpt_u[0].f_n is None:
            # If the density is initialized from atomic densities
            # the occupation numbers are not available yet
            # If the MOM reference orbitals are not initialized
            # (e.g. when the density is initialized from atomic
            # densities) set the occupation numbers according to
            # the supplied numbers
            return

        self.iters = 0
        self.f_sn_unique = self.find_unique_occupation_numbers()
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

        self.initialized = True

    def update_occupations(self):
        f_sn = np.zeros_like(self.numbers)

        if self.width != 0.0:
            if self.iters == 0:
                self.width_update_counter = 0
            if self.iters % self.niter_width_update == 0:
                self.gauss_width = self.width + self.width_update_counter \
                    * self.width_increment
                self.width_update_counter += 1

        for kpt in self.wfs.kpt_u:
            # Compute projections within equally occupied subspaces
            # and occupy orbitals with biggest projections
            for f_n_unique in self.f_sn_unique[kpt.s]:
                occupied = self.f_sn_unique[kpt.s][f_n_unique]
                n_occ = len(f_sn[kpt.s][occupied])
                unoccupied = f_sn[kpt.s] == 0

                P = np.zeros(len(f_sn[kpt.s]))
                # The projections are calculated only for orbitals
                # that have not already been occupied
                P[unoccupied] = self.calculate_mom_projections(kpt,
                                                               f_n_unique,
                                                               unoccupied)
                P_max = np.argpartition(P, -n_occ)[-n_occ:]
                f_sn[kpt.s][P_max] = f_n_unique

            self.numbers[kpt.s] = f_sn[kpt.s].copy()

            if self.width != 0.0:
                orbs, f_sn_gs = self.find_hole_and_excited_orbitals(f_sn, kpt)
                if orbs:
                    for o in orbs:
                        mask, gauss = self.gaussian_smearing(kpt,
                                                             f_sn_gs,
                                                             o,
                                                             self.gauss_width)
                        f_sn_gs[mask] += (o[1] * gauss)
                    f_sn[kpt.s] = f_sn_gs.copy()

        return f_sn

    def calculate_mom_projections(self, kpt, f_n_unique, unoccupied):
        if self.wfs.mode == 'lcao':
            P = np.dot(self.c_ref[kpt.s][f_n_unique].conj(),
                       np.dot(kpt.S_MM, kpt.C_nM[unoccupied].T))
        else:
            # Pseudo wave function overlaps
            P = self.wfs.integrate(self.wf[kpt.s][f_n_unique][:],
                                   kpt.psit_nG[unoccupied][:], True)

            # PAW corrections
            P_corr = np.zeros_like(P)
            for a, p_a in self.p_an[kpt.s][f_n_unique].items():
                P_corr += np.dot(kpt.P_ani[a][unoccupied].conj(), p_a).T
            P_corr = np.ascontiguousarray(P_corr)
            self.wfs.gd.comm.sum(P_corr)

            # Sum pseudo wave and PAW contributions
            P += P_corr

        P = np.sum(P ** 2, axis=0)
        P = P ** 0.5

        return P

    def find_hole_and_excited_orbitals(self, f_sn, kpt):
        # Assume zero-width occupations for ground state
        ne = int(f_sn[kpt.s].sum())
        f_sn_gs = np.zeros_like(f_sn[kpt.s])
        f_sn_gs[:ne] = 1.0
        f_sn_diff = f_sn[kpt.s] - f_sn_gs

        # Select hole and excited orbitals
        idxs = np.where(np.abs(f_sn_diff) > 1e-5)[0]
        w = f_sn_diff[np.abs(f_sn_diff) > 1e-5]
        orbs = list(zip(idxs, w))

        return orbs, f_sn_gs

    def gaussian_smearing(self, kpt, f_sn_gs, o, gauss_width):
        if o[1] < 0:
            mask = (f_sn_gs > 1e-8)
        else:
            mask = (f_sn_gs < 1e-8)

        e = kpt.eps_n[mask]
        de2 = -(e - kpt.eps_n[o[0]]) ** 2
        gauss = (1 / (gauss_width * np.sqrt(2 * np.pi)) *
                 np.exp(de2 / (2 * gauss_width ** 2)))
        gauss /= sum(gauss)

        return mask, gauss

    def find_unique_occupation_numbers(self):
        f_sn_unique = {}
        for kpt in self.wfs.kpt_u:
            f_sn_unique[kpt.s] = {}
            f_n = self.numbers[kpt.s]

            for f_n_unique in np.unique(f_n):
                if f_n_unique >= 1.0e-10:
                    f_sn_unique[kpt.s][f_n_unique] = f_n == f_n_unique

        return f_sn_unique
