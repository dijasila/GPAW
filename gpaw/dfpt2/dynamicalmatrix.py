import numpy as np

from ase.utils.timing import timer

from gpaw.utilities import unpack


class DynamicalMatrix:
    """Class for assembling the dynamical matrix from first-order responses.

    The second order derivative of the total energy with respect to atomic
    displacements (for periodic systems collective atomic displacements
    characterized by a q-vector) can be obtained from an expression involving
    the first-order derivatives of the density and the wave-functions.

    Each of the various contributions to the second order derivative of the
    total energy are implemented in separate functions.
    """
    def __init__(self, atoms, dtype=float, timer=None):
        """Initialize class with a list of atoms.

        Parameters
        ----------
        atoms: Atoms
            List of atoms for the system.
        dtype: float or complex
            Molecules and Gamma-point calculations have real dynamical
            matrices, the rest has complex ones.
        timer:

        """

        self.atoms = atoms
        self.dtype = dtype
        self.timer = timer

        # List of atomic indices to be included (default is all atoms)
        self.indices = range(len(self.atoms))

        masses = atoms.get_masses()
        # Array with inverse sqrt of masses repeated to match shape of mode
        # arrays
        self.m_inv_av = np.repeat(masses[self.indices]**-0.5, 3)

        # Matrix of force constants -- dict of dicts in atomic indices
        self.C_aavv = [dict([(a_, np.zeros((3, 3), dtype=dtype))
                             for a_ in self.indices]) for a in self.indices]

        # Dynamical matrix (3N,3N)
        self.D_nn = None

        self.assembled = False

    @timer('Assemble')
    def assemble(self):
        """Assemble dynamical matrix from the force constants attribute.

        The elements of the dynamical matrix are given by::

            D_ij(q) = 1/(M_i + M_j) * C_ij(q) ,

        where i and j are collective atomic and cartesian indices.

        During the assembly, various symmetries of the dynamical matrix are
        enforced::

            1) Hermiticity
           ### 2) Acoustic sum-rule
           ### 3) D(q) = D*(-q)

        """
        # Number of atoms included
        N = len(self.indices)

        # Assemble matrix of force constants
        C_avav = np.zeros((3*N, 3*N), dtype=self.dtype)
        for i, a in enumerate(self.indices):
            for j, a_ in enumerate(self.indices):
                C_avav[3*i: 3*i + 3, 3*j: 3*j + 3] += self.C_aavv[a][a_]

        # Make C(q) Hermitian
        C_avav *= 0.5
        C_avav += C_avav.conj().T

        # Mass prefactor for the dynamical matrix
        M_avav = self.m_inv_av[:, np.newaxis] * self.m_inv_av

        self.D_nn = C_avav * M_avav

        self.assembled = True

    @timer('Calculate row')
    def calculate_row(self, perturbation, response_calc):
        """Calculate row of force constant matrix from first-order derivatives.

        Parameters
        ----------
        perturbation: PhononPerturbation
            The perturbation which holds the derivative of the
            pseudo-potential.
        response_calc: ResponseCalculator
            Calculator with the corresponding derivatives of the density and
            the wave-functions.

        """
        self.density_derivative(perturbation, response_calc)
        self.wfs_derivative(perturbation, response_calc)

    @timer('Density ground state')
    def density_ground_state(self, calc):
        """Contributions involving ground-state density.

        These terms contains second-order derivaties of the localized functions
        ghat and vbar. They are therefore diagonal in the atomic indices.
        """

        # ATTENTION This routine may be incomplete.
        # ATTENTION: The second derivative is missing L>0 parts!!!!

        # Use the GS LFC's to integrate with the ground-state quantities!

        # Compensation charge coefficients
        Q_aL = calc.density.Q_aL

        # Integral of Hartree potential times the second derivative of ghat
        d2ghat_aLvv = calc.density.ghat.dict_d2(zero=True)
        calc.density.ghat.second_derivative(calc.hamiltonian.vHt_g, d2ghat_aLvv)

        # Integral of electron density times the second derivative of vbar
        d2vbar_avv = calc.hamiltonian.vbar.dict_d2(zero=True)
        calc.hamiltonian.vbar.second_derivative(calc.density.nt_g, d2vbar_avv)

        # Matrix of force constants to be updated; q=-1 for Gamma calculation!
        for a in self.indices:
            # HACK: HGH has only one ghat per atom -> generalize when
            # implementing PAW
            self.C_aavv[a][a] += d2ghat_aLvv[a][0] * Q_aL[a]
            # here the 0 is actually correct, because d2vbar doesn't depend on L
            self.C_aavv[a][a] += d2vbar_avv[a][0]

    @timer('Density derivative')
    def density_derivative(self, perturbation, response_calc):
        """Contributions involving the first-order density derivative."""

        # ATTENTION This routine has not been checked.

        # Get attributes from the phononperturbation
        a = perturbation.a
        v = perturbation.v
        # XXX careful here, Gamma calculation has q=-1
        q = perturbation.q

        # Localized functions
        ghat = perturbation.ghat
        vbar = perturbation.vbar
        # Compensation charge coefficients
        Q_aL = perturbation.Q_aL

        # Density derivative
        nt1_g = response_calc.nt1_g

        # Hartree potential derivative including compensation charges
        vH1_g = response_calc.vH1_g.copy()
        vH1_g += perturbation.vghat1_g

        # Integral of Hartree potential derivative times ghat derivative
        dghat_aLv = ghat.dict(derivative=True)
        # Integral of density derivative times vbar derivative
        dvbar_av = vbar.dict(derivative=True)

        # Evaluate integrals
        ghat.derivative(vH1_g, dghat_aLv, q=q)
        vbar.derivative(nt1_g, dvbar_av, q=q)

        # Add to force constant matrix attribute
        for a_ in self.indices:
            # Minus sign comes from lfc member function derivative
            self.C_aavv[a][a_][v] -= np.dot(Q_aL[a_], dghat_aLv[a_])
            self.C_aavv[a][a_][v] -= dvbar_av[a_][0]

    @timer('Wave function derivative')
    def wfs_derivative(self, perturbation, response_calc):
        """Contributions from the non-local part of the PAW potential."""

        # ATTENTION This routine has not been checked.

        # Get attributes from the phononperturbation
        a = perturbation.a
        v = perturbation.v
        # q = perturbation.q

        # Projector functions
        pt = response_calc.wfs.pt
        # Projector coefficients
        dH_asp = perturbation.dH_asp

        # K-point
        kpt_u = response_calc.wfs.kpt_u
        nbands = response_calc.nbands

        # Get k+q indices
        if perturbation.has_q():
            q_c = perturbation.get_q()
            kplusq_k = response_calc.wfs.kd.find_k_plus_q(q_c)
        else:
            kplusq_k = range(len(kpt_u))

        for kpt in kpt_u:
            # Indices of k and k+q
            k = kpt.k
            kplusq = kplusq_k[k]

            # Projector coefficients
            P_ani = kpt.P_ani
            dP_aniv = kpt.dP_aniv

            # psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # Overlap between wave-function derivative and projectors
            Pdpsi_ani = pt.dict(shape=nbands, zero=True)
            pt.integrate(psit1_nG, Pdpsi_ani, q=kplusq)
            # Overlap between wave-function derivative and derivative of
            # projectors
            dPdpsi_aniv = pt.dict(shape=nbands, derivative=True)
            pt.derivative(psit1_nG, dPdpsi_aniv, q=kplusq)

            for a_ in self.indices:
                # Coefficients from atom a
                Pdpsi_ni = Pdpsi_ani[a]
                dPdpsi_niv = -1 * dPdpsi_aniv[a]
                # Coefficients from atom a_
                H_ii = unpack(dH_asp[a_][0])
                P_ni = P_ani[a_]
                dP_niv = -1 * dP_aniv[a_]

                # Term with dPdpsi and P coefficients
                HP_ni = np.dot(P_ni, H_ii)
                dPdpsiHP_nv = (dPdpsi_niv.conj() *
                               HP_ni[:, :, np.newaxis]).sum(1)
                dPdpsiHP_nv *= kpt.weight
                A_v = dPdpsiHP_nv.sum(0)

                # Term with dP and Pdpsi coefficients
                HPdpsi_ni = np.dot(Pdpsi_ni.conj(), H_ii)
                dPHPdpsi_nv = (dP_niv * HPdpsi_ni[:, :, np.newaxis]).sum(1)
                dPHPdpsi_nv *= kpt.weight
                B_v = dPHPdpsi_nv.sum(0)

                # Factor of 2 from time-reversal symmetry
                C_ = 2 * (A_v + B_v)
                if self.dtype == complex:
                    self.C_aavv[a][a_][v] += C_
                else:
                    self.C_aavv[a][a_][v] += C_.real

    def get_mass_array(self):
        """Return inverse sqrt of masses (matches shape of mode array)."""

        assert self.m_inv_av is not None
        return self.m_inv_av

    def get_indices(self):
        """Return indices of included atoms."""

        return self.indices

    def acoustic_sum_rule(self, D_nn):
        """Applies acoustic sum rules to q=0 dynamical matrix.

        Currently this is only a simple version for the 3 translational degrees
        of freedom of crystals.

        """

        # NOTE All this needs to be done much nicer

        D_new = D_nn.copy()
        for i in range(3):
            for j in range(3):
                for ni in self.indices:
                    a = 3*ni + i
                    tmp = 0
                    for nj in self.indices:
                        b = 3*nj + j
                        if ni != nj:
                            tmp += D_nn[a, b]
                    D_new[a, 3*ni+j] = -tmp

        return D_new
