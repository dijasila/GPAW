import warnings
from math import sqrt, pi
import numpy as np

from ase.units import Ha
from gpaw.mpi import world
from gpaw.density import redistribute_array, redistribute_atomic_matrices
from gpaw.sphere.lebedev import weight_n
from gpaw.utilities import pack, pack_atomic_matrices, unpack_atomic_matrices
from gpaw.xc.gllb import safe_sqr
from gpaw.xc.gllb.contribution import Contribution

# XXX Work in process
debug = False


def d(*args):
    if debug:
        print(args)


class DiscontinuityPotential:
    """Container for discontinuity potential"""
    def __init__(self, response, vt_sG, Dresp_asp, D_asp):
        self.response = response
        self.vt_sG = vt_sG
        self.D_asp = D_asp
        self.Dresp_asp = Dresp_asp

    def redistribute(self, new_response):
        old_response = self.response
        new_wfs = new_response.wfs
        new_density = new_response.density
        self.vt_sG = redistribute_array(self.vt_sG,
                                        old_response.density.gd,
                                        new_density.gd,
                                        new_wfs.nspins,
                                        new_wfs.kptband_comm)

        def redist_asp(D_asp):
            return redistribute_atomic_matrices(D_asp,
                                                new_density.gd,
                                                new_wfs.nspins,
                                                new_density.setups,
                                                new_density.atom_partition,
                                                new_wfs.kptband_comm)

        self.D_asp = redist_asp(self.D_asp)
        self.Dresp_asp = redist_asp(self.Dresp_asp)
        self.response = new_response


class C_Response(Contribution):
    def __init__(self, weight, coefficients, damp=1e-10):
        Contribution.__init__(self, weight)
        d('In c_Response __init__', self)
        self.coefficients = coefficients
        self.vt_sg = None
        self.vt_sG = None
        self.nt_sG = None
        self.D_asp = None
        self.Dresp_asp = None
        self.Drespdist_asp = None
        self.just_read = False
        self.damp = damp

    def set_damp(self, damp):
        self.damp = damp

    def get_name(self):
        return 'RESPONSE'

    def get_desc(self):
        return self.coefficients.get_description()

    def initialize(self, density, hamiltonian, wfs):
        Contribution.initialize(self, density, hamiltonian, wfs)
        self.coefficients.initialize(wfs)
        if self.Dresp_asp is None:
            assert self.density.D_asp is None

    def initialize_1d(self, ae):
        Contribution.initialize_1d(self, ae)
        self.coefficients.initialize_1d(ae)

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        w_i = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g += self.weight * np.dot(w_i, u2_j) / (
            np.dot(self.ae.f_j, u2_j) + self.damp)
        return 0.0  # Response part does not contribute to energy

    def update_potentials(self, nt_sg):
        d('In update response potential')
        if self.just_read:
            # This is very hackish.
            # Reading GLLB-SC loads all density matrices to all cores.
            # This code first removes them to match density.D_asp.
            # and then further distributes the density matricies for
            # xc-corrections.
            d('Just read')

            Dresp_asp = self.empty_atomic_matrix()
            D_asp = self.empty_atomic_matrix()
            for a in self.density.D_asp:
                Dresp_asp[a][:] = self.Dresp_asp[a]
                D_asp[a][:] = self.D_asp[a]
            self.Dresp_asp = Dresp_asp
            self.D_asp = D_asp

            assert len(self.Dresp_asp) == len(self.density.D_asp)
            self.just_read = False
            self.Drespdist_asp = self.distribute_Dresp_asp(self.Dresp_asp)
            d('Core ', world.rank, 'self.Dresp_asp', self.Dresp_asp.items(),
              'self.Drespdist_asp', self.Drespdist_asp.items())
            self.Ddist_asp = self.distribute_Dresp_asp(self.D_asp)
            return

        # if not self.occupations.ready:
        if self.wfs.fermi_levels is None:
            d('No occupations calculated yet')
            return

        w_kn = self.coefficients.get_coefficients(self.wfs.kpt_u)
        f_kn = [kpt.f_n for kpt in self.wfs.kpt_u]
        if w_kn is not None:
            self.vt_sG = self.gd.zeros(self.nspins)
            self.nt_sG = self.gd.zeros(self.nspins)

            for kpt, w_n in zip(self.wfs.kpt_u, w_kn):
                self.wfs.add_to_density_from_k_point_with_occupation(
                    self.vt_sG, kpt, w_n)
                self.wfs.add_to_density_from_k_point(self.nt_sG, kpt)

            self.wfs.kptband_comm.sum(self.nt_sG)
            self.wfs.kptband_comm.sum(self.vt_sG)

            if self.wfs.kd.symmetry:
                for nt_G, vt_G in zip(self.nt_sG, self.vt_sG):
                    self.wfs.kd.symmetry.symmetrize(nt_G, self.gd)
                    self.wfs.kd.symmetry.symmetrize(vt_G, self.gd)

            d('response update D_asp', world.rank, self.Dresp_asp.keys(),
              self.D_asp.keys())
            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.Dresp_asp, w_kn)
            self.Drespdist_asp = self.distribute_Dresp_asp(self.Dresp_asp)
            d('response update Drespdist_asp', world.rank,
              self.Dresp_asp.keys(), self.D_asp.keys())
            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.D_asp, f_kn)
            self.Ddist_asp = self.distribute_Dresp_asp(self.D_asp)

            self.vt_sG /= self.nt_sG + self.damp

        self.vt_sg = self.finegd.zeros(self.nspins)
        self.density.distribute_and_interpolate(self.vt_sG, self.vt_sg)

    def calculate_spinpaired(self, e_g, n_g, v_g):
        self.update_potentials([n_g])
        v_g[:] += self.weight * self.vt_sg[0]
        return 0.0

    def calculate_spinpolarized(self, e_g, n_sg, v_sg):
        self.update_potentials(n_sg)
        v_sg += self.weight * self.vt_sg
        return 0.0

    def distribute_Dresp_asp(self, Dresp_asp):
        d('distribute_Dresp_asp')
        return self.density.atomdist.to_work(Dresp_asp)

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a,
                                         addcoredensity=True):
        # Get the XC-correction instance
        c = setup.xc_correction
        ncresp_g = setup.extra_xc_data['core_response'] / self.nspins
        if not addcoredensity:
            ncresp_g[:] = 0.0

        for D_p, dEdD_p, Dresp_p in zip(self.Ddist_asp[a], H_sp,
                                        self.Drespdist_asp[a]):
            D_Lq = np.dot(c.B_pqL.T, D_p)
            n_Lg = np.dot(D_Lq, c.n_qg)  # Construct density
            if addcoredensity:
                n_Lg[0] += c.nc_g * sqrt(4 * pi) / self.nspins
            nt_Lg = np.dot(
                D_Lq, c.nt_qg
            )  # Construct smooth density (_without smooth core density_)

            Dresp_Lq = np.dot(c.B_pqL.T, Dresp_p)
            nresp_Lg = np.dot(Dresp_Lq, c.n_qg)  # Construct 'response density'
            nrespt_Lg = np.dot(
                Dresp_Lq, c.nt_qg
            )  # Construct smooth 'response density' (w/o smooth core)

            for w, Y_L in zip(weight_n, c.Y_nL):
                nt_g = np.dot(Y_L, nt_Lg)
                nrespt_g = np.dot(Y_L, nrespt_Lg)
                x_g = nrespt_g / (nt_g + self.damp)
                dEdD_p -= self.weight * w * np.dot(
                    np.dot(c.B_pqL, Y_L), np.dot(c.nt_qg, x_g * c.rgd.dv_g))

                n_g = np.dot(Y_L, n_Lg)
                nresp_g = np.dot(Y_L, nresp_Lg)
                x_g = (nresp_g + ncresp_g) / (n_g + self.damp)

                dEdD_p += self.weight * w * np.dot(
                    np.dot(c.B_pqL, Y_L), np.dot(c.n_qg, x_g * c.rgd.dv_g))

        return 0.0

    def integrate_sphere(self, a, Dresp_sp, D_sp, Dwf_p, spin=0):
        c = self.wfs.setups[a].xc_correction
        Dresp_p, D_p = Dresp_sp[spin], D_sp[spin]
        D_Lq = np.dot(c.B_pqL.T, D_p)
        n_Lg = np.dot(D_Lq, c.n_qg)  # Construct density
        n_Lg[0] += c.nc_g * sqrt(4 * pi) / len(D_sp)
        nt_Lg = np.dot(D_Lq, c.nt_qg
                       )  # Construct smooth density (without smooth core)
        Dresp_Lq = np.dot(c.B_pqL.T, Dresp_p)  # Construct response
        nresp_Lg = np.dot(Dresp_Lq, c.n_qg)  # Construct 'response density'
        nrespt_Lg = np.dot(
            Dresp_Lq, c.nt_qg
        )  # Construct smooth 'response density' (w/o smooth core)
        Dwf_Lq = np.dot(c.B_pqL.T, Dwf_p)  # Construct lumo wf
        nwf_Lg = np.dot(Dwf_Lq, c.n_qg)
        nwft_Lg = np.dot(Dwf_Lq, c.nt_qg)
        E = 0.0
        for w, Y_L in zip(weight_n, c.Y_nL):
            v = np.dot(Y_L, nwft_Lg) * np.dot(Y_L, nrespt_Lg) / (
                np.dot(Y_L, nt_Lg) + self.damp)
            E -= self.weight * w * np.dot(v, c.rgd.dv_g)
            v = np.dot(Y_L, nwf_Lg) * np.dot(Y_L, nresp_Lg) / (
                np.dot(Y_L, n_Lg) + self.damp)
            E += self.weight * w * np.dot(v, c.rgd.dv_g)
        return E

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        v_g = np.zeros(self.ae.N)
        n_g = np.zeros(self.ae.N)
        for w_n, f_n, u_n in zip(w_ln, self.ae.f_ln,
                                 self.ae.s_ln):  # For each angular momentum
            u2_n = safe_sqr(u_n)
            v_g += np.dot(w_n, u2_n)
            n_g += np.dot(f_n, u2_n)

        vt_g += self.weight * v_g / (n_g + self.damp)
        return 0.0  # Response part does not contribute to energy

    def calculate_delta_xc(self, homolumo=None):
        warnings.warn(
            'The function calculate_delta_xc() is deprecated. '
            'Please use calculate_discontinuity_potential() instead. '
            'See documentation on calculating band gap with GLLBSC.',
            DeprecationWarning)

        if homolumo is None:
            # Calculate band gap
            pass

            # This always happens, so we don't warn.
            # We should perhaps print it as an ordinary message,
            # but we do not have a file here to which to print.
            # import warnings
            # warnings.warn('Calculating KS-gap directly from the k-points, '
            #               'can be inaccurate.')
            # homolumo = self.occupations.get_homo_lumo(self.wfs)
        Dxc = self.calculate_discontinuity_potential(homolumo)
        self.Dxc_vt_sG = Dxc.vt_sG
        self.Dxc_D_asp = Dxc.D_asp
        self.Dxc_Dresp_asp = Dxc.Dresp_asp

    def calculate_discontinuity_potential(self, homolumo):
        r"""Calculate the derivative discontinuity potential.

        This function calculates $`\Delta_{xc}(r)`$ as given by
        Eq. (24) of https://doi.org/10.1103/PhysRevB.82.115106

        Parameters:

        homolumo: homo and lumo energies in atomic units

        Returns:

        Dxc: Discontinuity potential
        """
        Dxc_Dresp_asp = self.empty_atomic_matrix()
        Dxc_D_asp = self.empty_atomic_matrix()
        for a in self.density.D_asp:
            ni = self.wfs.setups[a].ni
            Dxc_Dresp_asp[a] = np.zeros((self.nspins, ni * (ni + 1) // 2))
            Dxc_D_asp[a] = np.zeros((self.nspins, ni * (ni + 1) // 2))

        # Calculate new response potential with LUMO reference
        w_kn = self.coefficients.get_coefficients_for_lumo_perturbation(
            self.wfs.kpt_u,
            homolumo=homolumo)
        f_kn = [kpt.f_n for kpt in self.wfs.kpt_u]

        Dxc_vt_sG = self.gd.zeros(self.nspins)
        nt_sG = self.gd.zeros(self.nspins)
        for kpt, w_n in zip(self.wfs.kpt_u, w_kn):
            self.wfs.add_to_density_from_k_point_with_occupation(Dxc_vt_sG,
                                                                 kpt, w_n)
            self.wfs.add_to_density_from_k_point(nt_sG, kpt)

        self.wfs.kptband_comm.sum(nt_sG)
        self.wfs.kptband_comm.sum(Dxc_vt_sG)

        if self.wfs.kd.symmetry:
            for nt_G, Dxc_vt_G in zip(nt_sG, Dxc_vt_sG):
                self.wfs.kd.symmetry.symmetrize(nt_G, self.gd)
                self.wfs.kd.symmetry.symmetrize(Dxc_vt_G, self.gd)

        Dxc_vt_sG /= nt_sG + self.damp

        self.wfs.calculate_atomic_density_matrices_with_occupation(
            Dxc_Dresp_asp, w_kn)
        self.wfs.calculate_atomic_density_matrices_with_occupation(
            Dxc_D_asp, f_kn)
        Dxc = DiscontinuityPotential(self, Dxc_vt_sG, Dxc_Dresp_asp, Dxc_D_asp)
        return Dxc

    def calculate_delta_xc_perturbation(self):
        warnings.warn(
            'The function calculate_delta_xc_perturbation() is deprecated. '
            'Please use calculate_discontinuity() instead. '
            'See documentation on calculating band gap with GLLBSC.',
            DeprecationWarning)
        Dxc = DiscontinuityPotential(self, self.Dxc_vt_sG, self.Dxc_Dresp_asp,
                                     self.Dxc_D_asp)
        if self.nspins != 1:
            ret = []
            for spin in range(self.nspins):
                ret.append(self.calculate_discontinuity(Dxc, spin=spin))
            return ret
        return self.calculate_discontinuity(Dxc)

    def calculate_delta_xc_perturbation_spin(self, s=0):
        warnings.warn(
            'The function calculate_delta_xc_perturbation_spin() is '
            'deprecated. Please use calculate_discontinuity_spin() instead. '
            'See documentation on calculating band gap with GLLBSC.',
            DeprecationWarning)
        Dxc = DiscontinuityPotential(self, self.Dxc_vt_sG, self.Dxc_Dresp_asp,
                                     self.Dxc_D_asp)
        return self.calculate_discontinuity(Dxc, spin=s)

    def calculate_discontinuity(self,
                                Dxc: DiscontinuityPotential,
                                spin: int = None):
        r"""Calculate the derivative discontinuity.

        This function evaluates the perturbation theory expression
        for the derivative discontinuity value as given by
        Eq. (25) of https://doi.org/10.1103/PhysRevB.82.115106

        Parameters:

        Dxc:
            Discontinuity potential.
        spin:
            Spin value.

        Returns:

        KSgap:
            Kohn-Sham gap in eV
        dxc:
            Derivative discontinuity in eV
        """
        if spin is None:
            if self.nspins != 1:
                raise ValueError('Specify spin for the discontinuity.')
            spin = 0

        homo, lumo = self.wfs.get_homo_lumo(spin)
        KSgap = lumo - homo

        # Find the lumo-orbital of this spin
        eps_n = self.wfs.kpt_qs[0][spin].eps_n
        assert self.wfs.bd.comm.size == 1
        lumo_n = (eps_n < self.wfs.fermi_level).sum()

        # Calculate the expectation value for all k points, and keep
        # the smallest energy value
        nt_G = self.gd.empty()
        min_energy = np.inf
        for u, kpt in enumerate(self.wfs.kpt_u):
            if kpt.s == spin:
                nt_G[:] = 0.0
                self.wfs.add_orbital_density(nt_G, kpt, lumo_n)
                E = 0.0
                for a in Dxc.D_asp:
                    D_sp = Dxc.D_asp[a]
                    Dresp_sp = Dxc.Dresp_asp[a]
                    P_ni = kpt.P_ani[a]
                    Dwf_p = pack(np.outer(P_ni[lumo_n].T.conj(),
                                          P_ni[lumo_n]).real)
                    E += self.integrate_sphere(a, Dresp_sp, D_sp, Dwf_p,
                                               spin=spin)
                E = self.gd.comm.sum(E)
                E += self.gd.integrate(nt_G * Dxc.vt_sG[spin])
                E += kpt.eps_n[lumo_n] - lumo
                min_energy = min(min_energy, E)

        # Take the smallest value over all distributed k points
        dxc = self.wfs.kd.comm.min(min_energy)
        return KSgap * Ha, dxc * Ha

    def calculate_discontinuity_from_average(self,
                                             Dxc: DiscontinuityPotential,
                                             spin: int,
                                             testing: bool = False):
        msg = ('This function estimates discontinuity by calculating '
               'the average of the discontinuity potential. '
               'Use only for testing and debugging.')
        if not testing:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

        # Calculate average of lumo reference response potential
        dxc = np.average(Dxc.vt_sG[spin])
        return dxc * Ha

    def initialize_from_atomic_orbitals(self, basis_functions):
        # Initialize 'response-density' and density-matrices
        # print('Initializing from atomic orbitals')
        self.Dresp_asp = self.empty_atomic_matrix()
        setups = self.wfs.setups

        for a in self.density.D_asp.keys():
            ni = setups[a].ni
            self.Dresp_asp[a] = np.zeros((self.nspins, ni * (ni + 1) // 2))

        self.D_asp = self.empty_atomic_matrix()
        f_asi = {}
        w_asi = {}

        for a in basis_functions.atom_indices:
            if setups[a].type == 'ghost':
                w_j = [0.0]
            else:
                w_j = setups[a].extra_xc_data['w_j']

            # Basis function coefficients based of response weights
            w_si = setups[a].calculate_initial_occupation_numbers(
                0, False,
                charge=0,
                nspins=self.nspins,
                f_j=w_j)
            # Basis function coefficients based on density
            f_si = setups[a].calculate_initial_occupation_numbers(
                0, False,
                charge=0,
                nspins=self.nspins)

            if a in basis_functions.my_atom_indices:
                self.Dresp_asp[a] = setups[a].initialize_density_matrix(w_si)
                self.D_asp[a] = setups[a].initialize_density_matrix(f_si)

            f_asi[a] = f_si
            w_asi[a] = w_si

        self.Drespdist_asp = self.distribute_Dresp_asp(self.Dresp_asp)
        self.Ddist_asp = self.distribute_Dresp_asp(self.D_asp)
        self.nt_sG = self.gd.zeros(self.nspins)
        basis_functions.add_to_density(self.nt_sG, f_asi)
        self.vt_sG = self.gd.zeros(self.nspins)
        basis_functions.add_to_density(self.vt_sG, w_asi)
        # Update vt_sG to correspond atomic response potential. This will be
        # used until occupations and eigenvalues are available.
        self.vt_sG /= self.nt_sG + self.damp
        self.vt_sg = self.finegd.zeros(self.nspins)
        self.density.distribute_and_interpolate(self.vt_sG, self.vt_sg)

    def get_extra_setup_data(self, extra_data):
        ae = self.ae
        njcore = ae.njcore
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        w_j = []
        for w_n in w_ln:
            for w in w_n:
                w_j.append(w)
        extra_data['w_j'] = w_j

        w_j = self.coefficients.get_coefficients_1d()
        x_g = np.dot(w_j[:njcore], safe_sqr(ae.u_j[:njcore]))
        x_g[1:] /= ae.r[1:] ** 2 * 4 * np.pi
        x_g[0] = x_g[1]
        extra_data['core_response'] = x_g

        # For debugging purposes
        w_j = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g = self.weight * np.dot(w_j, u2_j) / (
            np.dot(self.ae.f_j, u2_j) + self.damp)
        v_g[0] = v_g[1]
        extra_data['all_electron_response'] = v_g

        # Calculate Hardness of spherical atom, for debugging purposes
        l = [np.where(f < 1e-3, e, 1000)
             for f, e in zip(self.ae.f_j, self.ae.e_j)]
        h = [np.where(f > 1e-3, e, -1000)
             for f, e in zip(self.ae.f_j, self.ae.e_j)]
        lumo_e = min(l)
        homo_e = max(h)
        if lumo_e < 999:  # If there is unoccpied orbital
            w_j = self.coefficients.get_coefficients_1d_for_lumo_perturbation()
            v_g = self.weight * np.dot(w_j, u2_j) / (
                np.dot(self.ae.f_j, u2_j) + self.damp)
            e2 = [e + np.dot(u2 * v_g, self.ae.dr)
                  for u2, e in zip(u2_j, self.ae.e_j)]
            lumo_2 = min([np.where(f < 1e-3, e, 1000)
                          for f, e in zip(self.ae.f_j, e2)])
            # print('New lumo eigenvalue:', lumo_2 * 27.2107)
            self.hardness = lumo_2 - homo_e
            # print('Hardness predicted: %10.3f eV' %
            #       (self.hardness * 27.2107))

    def write(self, writer):
        """Writes response specific data."""
        wfs = self.wfs
        kpt_comm = wfs.kd.comm
        gd = wfs.gd

        # Write the pseudodensity on the coarse grid:
        shape = (wfs.nspins,) + tuple(gd.get_size_of_global_array())
        writer.add_array('gllb_pseudo_response_potential', shape)
        if kpt_comm.rank == 0:
            for vt_G in self.vt_sG:
                writer.fill(gd.collect(vt_G) * Ha)

        def pack(X0_asp):
            X_asp = wfs.setups.empty_atomic_matrix(
                wfs.nspins, wfs.atom_partition)
            # XXX some of the provided X0_asp contain strangely duplicated
            # elements.  Take only the minimal set:
            for a in X_asp:
                X_asp[a][:] = X0_asp[a]
            return pack_atomic_matrices(X_asp)

        writer.write(gllb_atomic_density_matrices=pack(self.D_asp))
        writer.write(gllb_atomic_response_matrices=pack(self.Dresp_asp))

    def empty_atomic_matrix(self):
        assert self.wfs.atom_partition is self.density.atom_partition
        return self.wfs.setups.empty_atomic_matrix(self.wfs.nspins,
                                                   self.wfs.atom_partition)

    def read(self, reader):
        r = reader.hamiltonian.xc
        wfs = self.wfs

        self.vt_sG = wfs.gd.empty(wfs.nspins)
        d('Reading vt_sG')
        self.gd.distribute(r.gllb_pseudo_response_potential / reader.ha,
                           self.vt_sG)
        self.vt_sg = self.density.finegd.zeros(wfs.nspins)
        self.density.distribute_and_interpolate(self.vt_sG, self.vt_sg)

        def unpack(D_sP):
            return unpack_atomic_matrices(D_sP, wfs.setups)

        # Read atomic density matrices and non-local part of hamiltonian:
        self.D_asp = unpack(r.gllb_atomic_density_matrices)
        self.Dresp_asp = unpack(r.gllb_atomic_response_matrices)

        # Dsp and Dresp need to be redistributed
        self.just_read = True

    def heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(self, olddens):
        from gpaw.density import redistribute_array
        self.vt_sg = redistribute_array(self.vt_sg,
                                        olddens.finegd, self.finegd,
                                        self.wfs.nspins, self.wfs.kptband_comm)
        self.vt_sG = redistribute_array(self.vt_sG,
                                        olddens.gd, self.gd,
                                        self.wfs.nspins, self.wfs.kptband_comm)
