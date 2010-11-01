"""This module provides a class for assembling the dynamical matrix."""

__all__ = ["DynamicalMatrix"]

from math import sqrt, pi
import os
import pickle

import numpy as np
import numpy.fft as fft
import numpy.linalg as la
import ase.units as units

from gpaw import debug
from gpaw.mpi import serial_comm
from gpaw.utilities import unpack, unpack2

class DynamicalMatrix:
    """Class for assembling the dynamical matrix from first-order responses.

    The second order derivative of the total energy with respect to atomic
    displacements (for periodic systems collective atomic displacemnts
    characterized by a q-vector) can be obtained from an expression involving
    the first-order derivatives of the density and the wave-functions.
    
    Each of the various contributions to the second order derivative of the
    total energy are implemented in separate functions.
    
    """
    
    def __init__(self, atoms, kd, dtype=float):
        """Inititialize class with a list of atoms.

        Parameters
        ----------
        atoms: Atoms
            List of atoms for the system.
        kd: KPointDescriptor
            Descriptor for the q-vector grid.

        """

        # Store useful objects
        self.atoms = atoms
        self.kd = kd
        self.dtype = dtype
        self.masses = atoms.get_masses()

        # Set default filename and path for the force constant files
        self.set_name_and_path()
        
        # List of atomic indices to be included (default is all atoms)
        self.indices = range(len(self.atoms))
        
        # Index of the gamma point -- for the acoustic sum-rule
        self.gamma_index = None
        
        if self.kd.gamma:
            self.gamma_index = 0
            assert dtype == float
        else:
            for k, k_c in enumerate(self.kd.ibzk_kc):
                if np.all(k_c == 0.):
                    self.gamma_index = k

        assert self.gamma_index is not None
        
        # Matrix of force constants -- dict of dicts in atomic indices
        # Only local q-vector stored here
        self.C_qaavv = [dict([(i, dict([(j, np.zeros((3, 3), dtype=dtype))
                                        for j in self.indices]))
                              for i in self.indices])
                        for q in range(self.kd.nibzkpts)]
        
        # Dynamical matrix -- 3Nx3N ndarray (vs q)
        # Local matrices
        self.D_q = []
        # Global array of matrices used upon collecting from slaves
        self.D_k = None
        # Dynamical matrix in the full Brillouin zone
        self.D = None
        
        self.assembled = False
        # XXX deprecated
        self.collected = False

    def __getstate__(self): 
        """Method used to pickle an instance of ``DynamicalMatrix``.

        Bound method attributes cannot be pickled and must therefore be deleted
        before an instance is dumped to file.

        """

        # Get state of object and take care of troublesome attributes
        state = dict(self.__dict__)
        state['kd'].__dict__['comm'] = serial_comm

        return state

    def __setstate__(self, state):
        """Method used to unpickle an instance of ``DynamicalMatrix``."""

        self.__dict__.update(state)
      
    def set_indices(self, indices):
        """Set indices of atoms to include in the calculation."""
        
        self.indices = indices

        self.C_qaavv = [dict([(a, dict([(a_, np.zeros((3, 3), dtype=dtype))
                                        for a_ in self.indices]))
                              for a in self.indices])
                        for q in range(self.kd.nibzkpts)]

    def set_name_and_path(self, name=None, path=None):
        """Set name and path of the force constant files.

        name: str
            Base name for the files which the elements of the matrix of force
            constants will be written to.
        path: str
            Path specifying the directory where the files will be dumped.
            
        """

        if name is None:
            self.name = self.atoms.get_name() + \
                        'D_nibzkpts_%i' % self.kd.nibzkpts
        else:
            self.name = name
            
        if path is None:
            self.path = '.'
        else:
            self.path = path
        
    def get_filename_string(self):
        """Return string template for filenames."""

        name_str = (self.name + '.' + 'q_%%0%ii_' % len(str(self.kd.nibzkpts)) +
                    'a_%%0%ii_' % len(str(len(self.atoms))) + 'v_%i' + '.pckl')

        return os.path.join(self.path, name_str)

    def write(self, q, a, v):
        """Write force constants for specified indices to file."""

        # Create filename
        filename_str = self.get_filename_string()
        filename = filename_str % (q, a, v)

        # Dump force constants
        fd = open(filename, 'w')
        C_qav_a = [self.C_qaavv[q][a][a_][v] for a_ in self.indices]
        pickle.dump(C_qav_a, fd)
        fd.close()
        
    def read(self):
        """Read the force constants from files."""

        filename_str = self.get_filename_string()

        for q in range(self.kd.nibzkpts):
            for a in self.indices:
                for v in [0, 1, 2]:
                    filename = filename_str % (q, a, v)
                    # print filename
                    fd = open(filename)
                    C_qav_a = pickle.load(fd)
                    fd.close()
                    for a_ in self.indices:
                        self.C_qaavv[q][a][a_][v] = C_qav_a[a_]

    def clean(self):
        """Delete generated files."""

        filename_str = self.get_filename_string()
        
        for q in range(self.kd.nibzkpts):
            for a in range(len(self.atoms)):
                for v in [0, 1, 2]:
                    filename = filename_str % (q, a, v)
                    if os.path.isfile(filename):
                        os.remove(filename)
                        
    def assemble(self, acoustic=True):
        """Assemble dynamical matrix from the force constants attribute.

        The elements of the dynamical matrix are given by::

            D_ij(q) = 1/(M_i + M_j) * C_ij(q) ,
                      
        where i and j are collective atomic and cartesian indices.

        During the assembly, various symmetries of the dynamical matrix are
        enforced::

            1) Hermiticity
            2) Acoustic sum-rule
            3) D(q) = D*(-q)

        Parameters
        ----------
        acoustic: bool
            When True, the diagonal of the matrix of force constants is
            corrected to ensure that the acoustic sum-rule is fulfilled.
            
        """

        # Read force constants from files
        self.read()

        # Number of atoms included
        N = len(self.indices)
        
        # Assemble matrix of force constants
        for q, C_aavv in enumerate(self.C_qaavv):

            C_avav = np.zeros((3*N, 3*N), dtype=self.dtype)
    
            for a, i in enumerate(self.indices):
                for a_, j in enumerate(self.indices):
                    C_avav[3*i : 3*i + 3, 3*j : 3*j + 3] += C_aavv[a][a_]

            self.D_q.append(C_avav)

        # XXX
        self.D_k = np.asarray(self.D_q)

        # XXX Figure out in which order the corrections should be done
        # Make C(q) Hermitian
        for C in self.D_k:
            C *= 0.5
            C += C.conj().T
            
        # Apply acoustic sum-rule if requested
        if acoustic:

            # Get matrix of force constants in the Gamma-point
            C_gamma = self.D_k[self.gamma_index].copy()

            # Correct atomic diagonal for each q-vector
            for C in self.D_k:

                for a in range(N):
                    C_gamma_av = C_gamma[3*a: 3*a+3]

                    for a_ in range(N):
                        C[3*a : 3*a + 3, 3*a : 3*a + 3] -= \
                              C_gamma_av[:3, 3*a_: 3*a_+3]

           
        # Move this bit to an ``unfold`` member function
        # XXX Time-reversal symmetry
        if self.kd.nibzkpts != self.kd.nbzkpts:
            self.D = np.concatenate((self.D_k[:0:-1].conjugate(), self.D_k))
        else:
            self.D = 0.5 * self.D_k
            self.D += self.D[::-1].conjugate()
            
        # Mass prefactor for the dynamical matrix
        m_av = np.repeat(np.asarray(self.masses)**(-0.5), 3)
        M_avav = m_av[:, np.newaxis] * m_av

        for C in self.D:
            C *= M_avav

        self.assembled = True
       
    def real_space(self):
        """Fourier transform the dynamical matrix to real-space."""

        if not self.assembled:
            self.assemble()

        # Shape of q-point grid
        N_c = tuple(self.kd.N_c)

        # Reshape before Fourier transforming
        shape = self.D.shape
        Dq_lmn = self.D.reshape(N_c + shape[1:])
        DR_lmn = fft.ifftn(fft.ifftshift(Dq_lmn, axes=(0, 1, 2)), axes=(0, 1, 2))

        if debug:
            # Check that D_R is real enough
            assert np.all(DR_lmn.imag < 1e-8)
            
        DR_lmn = DR_lmn.real

        # Corresponding R_m vectors in units of the lattice vectors
        N1_c = np.array(N_c)[:, np.newaxis]
        R_cm = np.indices(N1_c).reshape(3, -1)
        R_cm += N1_c // 2
        R_cm %= N1_c
        R_cm -= N1_c // 2

        return DR_lmn, R_cm
    
    def band_structure(self, path_kc, modes=False):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space matrix. In case of negative eigenvalues
        (squared frequency), the corresponding negative frequency is returned.

        Parameters
        ----------
        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
            
        """

        for k_c in path_kc:
            assert np.all(np.asarray(k_c) <= 1.0), \
                   "Scaled coordinates must be given"

        # Get the dynamical matrix in real-space
        DR_lmn, R_cm = self.real_space()

        # Reshape for the evaluation of the fourier sums
        shape = DR_lmn.shape
        DR_m = DR_lmn.reshape((-1,) + shape[-2:])
        
        # Lists for frequencies and modes along path
        omega_kn = []
        u_kn =  []
        
        for q_c in path_kc:

            # Evaluate fourier transform 
            phase_m = np.exp(-2.j * pi * np.dot(q_c, R_cm))
            D_q = np.sum(phase_m[:, np.newaxis, np.newaxis] * DR_m, axis=0)

            if modes:
                omega2_n, u_avn = la.eigh(D_q, UPLO='L')
                # Sort eigenmodes according to eigenvalues (see below)
                u_n = u_avn[:, omega2_n.argsort()]
                u_kn.append(u_avn)
            else:
                omega2_n = la.eigvalsh(D_q, UPLO='L')

            # Sort eigenvalues in increasing order
            omega2_n.sort()
            # Use dtype=complex to handle negative eigenvalues
            omega_n = np.sqrt(omega2_n.astype(complex))

            # Take care of imaginary frequencies
            if not np.all(omega2_n >= 0.):
                indices = np.where(omega2_n < 0)[0]
                print ("WARNING, %i imaginary frequencies at "
                       "q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)"
                       % (len(indices), q_c[0], q_c[1], q_c[2],
                          omega_n[indices][0].imag))
                
                omega_n[indices] = -1 * np.sqrt(np.abs(omega2_n[indices].real))

            omega_kn.append(omega_n.real)

        if modes:
            return np.asarray(omega_kn), np.asarray(u_kn)
        
        return np.asarray(omega_kn)
    
    def update_row(self, perturbation, response_calc):
        """Update row of force constant matrix from first-order derivatives.

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
        # XXX
        # self.wfs_derivative(perturbation, response_calc)
        
    def density_ground_state(self, calc):
        """Contributions involving ground-state density.

        These terms contains second-order derivaties of the localized functions
        ghat and vbar. They are therefore diagonal in the atomic indices.

        """

        # Use the GS LFC's to integrate with the ground-state quantities !
        ghat = calc.density.ghat
        vbar = calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = calc.density.Q_aL
        
        # Integral of Hartree potential times the second derivative of ghat
        vH_g = calc.hamiltonian.vHt_g
        d2ghat_aLvv = dict([(atom.index, np.zeros((3, 3)))
                            for atom in self.atoms])
        ghat.second_derivative(vH_g, d2ghat_aLvv)

        # Integral of electron density times the second derivative of vbar
        nt_g = calc.density.nt_g
        d2vbar_avv = dict([(atom.index, np.zeros((3, 3)))
                           for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        # Matrix of force constants to be updated; q=-1 for Gamma calculation!
        for C_aavv in self.C_qaavv:

            for a in self.indices:
                # XXX: HGH has only one ghat pr atoms -> generalize when
                # implementing PAW
                C_aavv[a][a] += d2ghat_aLvv[a] * Q_aL[a]
                C_aavv[a][a] += d2vbar_avv[a]

    def wfs_ground_state(self, calc, response_calc):
        """Ground state contributions from the non-local potential."""

        # Projector functions
        pt = calc.wfs.pt
        # Projector coefficients
        dH_asp = calc.hamiltonian.dH_asp
      
        # K-point
        kpt_u = response_calc.wfs.kpt_u
        nbands = response_calc.nbands
        
        for kpt in kpt_u:

            # Index of k
            k = kpt.k
            P_ani = kpt.P_ani
            dP_aniv = kpt.dP_aniv
            
            # Occupation factors include the weight of the k-points
            f_n = kpt.f_n
            psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # Calculate d2P_anivv coefficients
            # d2P_anivv = self.calculate_d2P_anivv()
            d2P_anivv = dict([(atom.index,
                               np.zeros(
                (nbands, pt.get_function_count(atom.index), 3, 3)
                )) for atom in self.atoms])
            #XXX Temp dict, second_derivative method only takes a_G array
            # -- no extra dims
            d2P_avv = dict([(atom.index, np.zeros((3, 3)))
                            for atom in self.atoms])
         
            for n in range(nbands):
                pt.second_derivative(psit_nG[n], d2P_avv)
                # Insert in other dict
                for atom in self.atoms:
                    a = atom.index
                    d2P_anivv[a][n, 0] = d2P_avv[a]
            
            for a in self.indices:
    
                H_ii = unpack(dH_asp[a][0])
                P_ni = P_ani[a]
                dP_niv = -1 * dP_aniv[a]
                d2P_nivv = d2P_anivv[a]
                
                # Term with second-order derivative of projector
                HP_ni = np.dot(P_ni, H_ii)
                d2PHP_nvv = (d2P_nivv.conj() *
                             HP_ni[:, :, np.newaxis, np.newaxis]).sum(1)
                assert False, "you are using the f_n attribute here"
                d2PHP_nvv *= kpt.f_n[:, np.newaxis, np.newaxis]
                A_vv = d2PHP_nvv.sum(0)
    
                # Term with first-order derivative of the projectors
                HdP_inv = np.dot(H_ii, dP_niv.conj())
                HdP_niv = np.swapaxes(HdP_inv, 0, 1)
                HdP_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
    
                B_vv = (dP_niv[:, :, np.newaxis, :] * 
                        HdP_niv[:, :, :, np.newaxis]).sum(0).sum(0)

                for C_aavv in self.C_qaavv:
                    
                    C_aavv[a][a] += (A_vv + B_vv) + (A_vv + B_vv).conj()

    def core_corrections(self):
        """Contribution from the derivative of the core density."""

        raise NotImplementedError
    
    def density_derivative(self, perturbation, response_calc):
        """Contributions involving the first-order density derivative."""

        # Get attributes from the phononperturbation
        a = perturbation.a
        v = perturbation.v
        #XXX careful here, Gamma calculation has q=-1
        q = perturbation.q

        # Matrix of force constants to be updated; q=-1 for Gamma calculation!
        C_aavv = self.C_qaavv[q]
        
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
            C_aavv[a][a_][v] -= np.dot(Q_aL[a_], dghat_aLv[a_])
            C_aavv[a][a_][v] -= dvbar_av[a_][0]

    def wfs_derivative(self, perturbation, response_calc):
        """Contributions from the non-local part of the PAW potential."""

        # Get attributes from the phononperturbation
        a = perturbation.a
        v = perturbation.v
        q = perturbation.q

        # Matrix of force constants to be updated
        C_aavv = self.C_qaavv[q]
           
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
            
            # Occupation factors include the weight of the k-points
            f_n = kpt.f_n
            psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # Overlap between wave-function derivative and projectors
            Pdpsi_ani = pt.dict(shape=nbands, zero=True)
            pt.integrate(psit1_nG, Pdpsi_ani, q=kplusq)
            # Overlap between wave-function derivative and derivative of projectors
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
                dPdpsiHP_nv = (dPdpsi_niv.conj() * HP_ni[:, :, np.newaxis]).sum(1)
                dPdpsiHP_nv *= f_n[:, np.newaxis]
                A_v = dPdpsiHP_nv.sum(0)
    
                # Term with dP and Pdpsi coefficients
                HPdpsi_ni = np.dot(Pdpsi_ni.conj(), H_ii)
                dPHPdpsi_nv = (dP_niv * HPdpsi_ni[:, :, np.newaxis]).sum(1)
                dPHPdpsi_nv *= f_n[:, np.newaxis]
                B_v = dPHPdpsi_nv.sum(0)

                # Factor of 2 from time-reversal symmetry
                C_aavv[a][a_][v] += 2 * (A_v + B_v)
