"""Module for calculations using the Maximum Overlap Method (MOM).
   https://arxiv.org/abs/2102.06542
   https://doi.org/10.1021/acs.jctc.0c00597.
"""

import numpy as np

from ase.units import Ha

from gpaw.occupations import FixedOccupationNumbers


def mom_calculation(calc,
                    atoms,
                    numbers,
                    update_fixed_occupations=True,
                    project_overlaps=True,
                    width=0.0,
                    width_increment=0.0,
                    niter_width_update=10):
    """Helper function to prepare a calculator for a MOM calculation.

       calc: GPAW instance
           GPAW calculator object.
       atoms: ASE instance
           ASE atoms object.
       numbers: list (len=nspins) of lists (len=nbands)
           Occupation numbers (in the range from 0 to 1) used to
           initialize the MOM reference orbitals.
       update_fixed_occupations: bool
           If True, the attribute 'numbers' gets updated with the
           calculated occupation numbers, such that when changing
           atomic positions the MOM reference orbitals will be
           initialized as the occupied orbitals found at convergence
           for the previous geometry. If False, when changing
           positions the MOM reference orbitals will be initialized
           from the orbitals of the previous geometry according to
           the user-supplied 'numbers'.
       project_overlaps: bool
           If True, the occupied orbitals at step k are chosen as
           the orbitals {|psi^(k)_m>} with the biggest weights
           P_m evaluated as the projections onto the manifold of
           reference orbitals {|psi_n>}:
           P_m = (Sum_n(|O_nm|^2))^0.5 (O_nm = <psi_n|psi^(k)_m>)
           See https://doi.org/10.1021/acs.jctc.7b00994.
           If False, the weights are evaluated as:
           P_m = max_n(|O_nm|)
           See https://doi.org/10.1021/acs.jctc.0c00488.
       width: float
           Width of Gaussian function in eV.
           See https://doi.org/10.1021/acs.jctc.0c00597.
       width_increment: float
       niter_width_update: int
    """

    if calc.wfs is None:
        # We need the wfs object to initialize OccupationsMOM
        calc.initialize(atoms)

    parallel_layout = calc.wfs.occupations.parallel_layout
    occ = FixedOccupationNumbers(numbers, parallel_layout)

    occ_mom = OccupationsMOM(calc.wfs,
                             occ,
                             numbers,
                             update_fixed_occupations,
                             project_overlaps,
                             width,
                             width_increment,
                             niter_width_update)
    calc.set(occupations=occ_mom)

    calc.log(occ_mom)


class OccupationsMOM:
    def __init__(self,
                 wfs,
                 occ,
                 numbers,
                 update_fixed_occupations=True,
                 project_overlaps=True,
                 width=0.0,
                 width_increment=0.0,
                 niter_width_update=10):
        self.wfs = wfs
        self.occ = occ
        self.extrapolate_factor = occ.extrapolate_factor
        self.numbers = np.array(numbers)
        self.update_fixed_occupations = update_fixed_occupations
        self.project_overlaps = project_overlaps
        self.width = width / Ha
        self.width_increment = width_increment / Ha
        self.niter_width_update = niter_width_update

        self.name = 'mom'
        self.iters = 0
        self.initialized = False

    def todict(self):
        dct = {'name': self.name,
               'numbers': self.numbers,
               'update_fixed_occupations': self.update_fixed_occupations,
               'project_overlaps': self.project_overlaps}
        if self.width != 0.0:
            dct['width'] = self.width * Ha
            dct['width_increment'] = self.width_increment * Ha
            dct['niter_width_update'] = self.niter_width_update
        return dct

    def __str__(self):
        s = 'Excited-state calculation with Maximum Overlap Method\n'
        s += '  Gaussian smearing of holes and excited electrons: '
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
            # If MOM reference orbitals are not initialized yet (e.g. when
            # the calculation is initialized from atomic densities), update
            # the occupation numbers according to the user-supplied numbers
            self.occ.f_sn = self.numbers.copy()
            self.initialize_reference_orbitals()
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
            # If the occupation numbers are not available (e.g. when
            # the calculation is initialized from atomic densities)
            # we first need to take a step of eigensolver and update
            # the occupation numbers according to the user-supplied
            # numbers before initializing the MOM reference orbitals
            return

        self.iters = 0
        # Initialize MOM reference orbitals for each equally
        # occupied subspace separately
        self.f_sn_unique = self.find_unique_occupation_numbers()
        if self.wfs.mode == 'lcao':
            self.c_ref = {}
            for kpt in self.wfs.kpt_u:
                self.c_ref[kpt.s] = {}
                for f_n_unique in self.f_sn_unique[kpt.s]:
                    occupied = self.f_sn_unique[kpt.s][f_n_unique]
                    self.c_ref[kpt.s][f_n_unique] = kpt.C_nM[occupied].copy()
        else:
            self.wf = {}
            self.p_an = {}
            for kpt in self.wfs.kpt_u:
                self.wf[kpt.s] = {}
                self.p_an[kpt.s] = {}
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
                P[unoccupied] = self.calculate_weights(kpt,
                                                       f_n_unique,
                                                       unoccupied)
                P_max = np.argpartition(P, -n_occ)[-n_occ:]
                f_sn[kpt.s][P_max] = f_n_unique

            if self.update_fixed_occupations:
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

    def calculate_weights(self, kpt, f_n_unique, unoccupied):
        if self.wfs.mode == 'lcao':
            O = np.dot(self.c_ref[kpt.s][f_n_unique].conj(),
                       np.dot(kpt.S_MM, kpt.C_nM[unoccupied].T))
        else:
            # Pseudo wave function overlaps
            O = self.wfs.integrate(self.wf[kpt.s][f_n_unique][:],
                                   kpt.psit_nG[unoccupied][:], True)

            # PAW corrections
            O_corr = np.zeros_like(O)
            for a, p_a in self.p_an[kpt.s][f_n_unique].items():
                O_corr += np.dot(kpt.P_ani[a][unoccupied].conj(), p_a).T
            O_corr = np.ascontiguousarray(O_corr)
            self.wfs.gd.comm.sum(O_corr)

            # Sum pseudo wave and PAW contributions
            O += O_corr

        if self.project_overlaps:
            #TODO: Replace 'O' with 'abs(O)'
            P = np.sum(O ** 2, axis=0)
            P = P ** 0.5
        else:
            P = np.amax(abs(O), axis=0)

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
