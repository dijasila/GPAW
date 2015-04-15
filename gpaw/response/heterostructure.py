from __future__ import print_function

import pickle

import numpy as np
from ase.units import Hartree, Bohr


class Heterostructure:
    def __init__(self, q_abs, frequencies, d,
                 chi_monopole, z, drho_monopole, d0=None, chi_dipole=None,
                 drho_dipole=None, layer_indices=None):
        # layers and distances
        self.n_types = len(chi_monopole)
        self.n_layers = len(d) + 1
        self.d = d / Bohr  # interlayer distances
        # space around each layer
        if len(d) > 0:
            self.s = (np.insert(self.d, 0, self.d[0]) + \
                      np.append(self.d, self.d[-1])) / 2.
        else:  # Monolayer calculation
            self.s = [d0 / Bohr]  # Width of layers

        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = range(self.n_layers)

        self.dim = self.n_layers
        if chi_dipole is not None:
            self.dim *= 2

        # Grid stuff
        self.z = z
        self.poisson_lim = 100  # above this limit use potential model
        system_size = np.sum(self.d) + 50
        self.z_lim = system_size
        self.dz = 0.01
        self.z_big = np.arange(0, self.z_lim, self.dz) - 25  # master grid
        self.z0 = np.append(np.array([0]), np.cumsum(self.d))

        # layer quantities
        self.q_abs = q_abs
        self.frequencies = frequencies
        self.chi_monopole = np.array(chi_monopole)
        self.chi_dipole = chi_dipole
        if chi_dipole is not None:
            self.chi_dipole = np.array(chi_dipole)
        
        # arange potential and density
        self.drho_monopole, self.drho_dipole, self.basis_array, \
            self.drho_array = self.arange_basis(drho_monopole, drho_dipole)
       
        self.dphi_array = self.get_induced_potentials()
        self.kernel_qij = None

    def arange_basis(self, drhom, drhod=None):
        from scipy.interpolate import interp1d
        Nz = len(self.z_big)
        drho_array = np.zeros([self.dim, len(self.q_abs),
                               Nz], dtype=complex)
        basis_array = np.zeros([self.dim, len(self.q_abs),
                                Nz], dtype=complex)
        
        for i in range(self.n_types):
            z = self.z[i] - self.z[i][len(self.z[i]) / 2]
            drhom_i = drhom[i]
            fm = interp1d(z, np.real(drhom_i))
            fm2 = interp1d(z, np.imag(drhom_i))
            if drhod is not None:
                drhod_i = drhod[i]
                fd = interp1d(z, np.real(drhod_i))
                fd2 = interp1d(z, np.imag(drhod_i))
            for k in [k for k in range(self.n_layers) \
                          if self.layer_indices[k] == i]:
                z_big = self.z_big - self.z0[k]
                i_1s = np.argmin(np.abs(-self.s[i] / 2. - z_big))
                i_2s = np.argmin(np.abs(self.s[i] / 2. - z_big))

                i_1 = np.argmin(np.abs(z[0] - z_big)) + 1
                i_2 = np.argmin(np.abs(z[-1] - z_big)) - 1
                if drhod is not None:
                    drho_array[2 * k, :, i_1: i_2] = \
                        fm(z_big[i_1: i_2]) + 1j * fm2(z_big[i_1: i_2])
                    basis_array[2 * k, :, i_1s: i_2s] = 1. / self.s[i]
                    drho_array[2 * k + 1, :, i_1: i_2] = \
                        fd(z_big[i_1: i_2]) + 1j * fd2(z_big[i_1: i_2])
                    basis_array[2 * k + 1, :, i_1: i_2] = \
                        fd(z_big[i_1: i_2]) + 1j * fd2(z_big[i_1: i_2])
                    #basis_array[2 * k + 1, i_1s: i_2s] = z_big[i_1s: i_2s] \
                    #    / self.s[i]**3 * 12.
                else:
                    drho_array[k, :, i_1: i_2] = \
                        fm(z_big[i_1: i_2]) + 1j * fm2(z_big[i_1: i_2])
                    basis_array[k, i_1s: i_2s] = 1. / self.s[i]
        
        return drhom, drhod, basis_array, drho_array

    def get_induced_potentials(self):
        from scipy.interpolate import interp1d
        Nz = len(self.z_big)
        dphi_array = np.zeros([self.dim, len(self.q_abs), Nz], dtype=complex)

        for i in range(self.n_types):
            z = self.z[i]
            for iq in range(len(self.q_abs)):
                q = self.q_abs[iq]
                drho_m = self.drho_monopole[i][iq].copy()
                poisson_m = self.solve_poisson_1D(drho_m, q, z)
                z_poisson = self.get_z_grid(z, z_lim=self.poisson_lim)
                fm = interp1d(z_poisson, np.real(poisson_m))
                fm2 = interp1d(z_poisson, np.imag(poisson_m))
                if self.chi_dipole is not None:
                    drho_d = self.drho_dipole[i][iq].copy()
                    #  delta = distance bewteen dipole peaks / 2
                    delta = np.abs(z[np.argmax(drho_d)] - \
                                   z[np.argmin(drho_d)]) / 2.
                    poisson_d = self.solve_poisson_1D(drho_d, q, z,
                                                      dipole=True,
                                                      delta=delta)
                    fd = interp1d(z_poisson, np.real(poisson_d))

                for k in [k for k in range(self.n_layers) \
                              if self.layer_indices[k] == i]:
                    z_big = self.z_big - self.z0[k]
                    i_1 = np.argmin(np.abs(z_poisson[0] - z_big)) + 1
                    i_2 = np.argmin(np.abs(z_poisson[-1] - z_big)) - 1

                    dphi_array[self.dim / self.n_layers * k, iq] = \
                        self.potential_model(self.q_abs[iq], self.z_big,
                                             self.z0[k])
                    dphi_array[self.dim / self.n_layers * k, iq, i_1: i_2] = \
                        fm(z_big[i_1: i_2]) + 1j * fm2(z_big[i_1: i_2])
                    if self.chi_dipole is not None:
                        dphi_array[2 * k + 1, iq] = \
                            self.potential_model(self.q_abs[iq], self.z_big,
                                                 self.z0[k], dipole=True,
                                                 delta=delta)
                        dphi_array[2 * k + 1, iq, i_1: i_2] = \
                            fd(z_big[i_1: i_2])
        
        return dphi_array

    def get_z_grid(self, z, z_lim=None):
        dz = z[1] - z[0]
        if z_lim is None:
            z_lim = self.z_lim
        z_lim = int(z_lim / dz) * dz

        z_grid = np.insert(z, 0, np.arange(-z_lim, z[0], dz))
        z_grid = np.append(z_grid, np.arange(z[-1] + dz, z_lim, dz))
        return z_grid
    
    def potential_model(self, q, z, z0=0, dipole=False, delta=None):
        """
        2D Coulomb: 2 pi / q with exponential decay in z-direction
        """
        if dipole:  # Two planes separated by 2*delta
            V = np.pi / (q * delta) * \
                (-np.exp(-q * np.abs(z - z0 + delta)) + \
                      np.exp(-q * np.abs(z - z0 - delta)))
        else:  # Monopole potential from single plane
            V = 2 * np.pi / q * np.exp(-q * np.abs(z - z0))
        
        return V
    
    def solve_poisson_1D(self, drho, q, z,
                         dipole=False, delta=None):
        
        z -= z[len(z) / 2]  # center arround 0
        z_grid = self.get_z_grid(z, z_lim=self.poisson_lim)
        dz = z[1] - z[0]
        Nz_loc = (len(z_grid) - len(z)) / 2
       
        drho = np.append(np.insert(drho, 0, np.zeros([Nz_loc])),
                         np.zeros([Nz_loc]))
        Nint = len(drho) - 1
        
        bc_v0 = self.potential_model(q, z_grid[0], dipole=dipole,
                                     delta=delta)
        bc_vN = self.potential_model(q, z_grid[-1], dipole=dipole,
                                     delta=delta)
        M = np.zeros((Nint + 1, Nint + 1))
        f_z = np.zeros(Nint + 1, dtype=complex)
        f_z[:] = - 4 * np.pi * drho[:]
        # Finite Difference Matrix
        for i in range(1, Nint):
            M[i, i] = -2. / (dz**2) - q**2
            M[i, i + 1] = 1. / dz**2
            M[i, i - 1] = 1. / dz**2
            M[0, 0] = 1.
            M[Nint, Nint] = 1.
    
        f_z[0] = bc_v0
        f_z[Nint] = bc_vN

        # Getting the Potential
        M_inv = np.linalg.inv(M)
        dphi = np.dot(M_inv, f_z)  # -q**2 * np.dot(M_inv,f_z)

        return dphi
    
    """
    # Offdiagonal elements of chi_tilde
    def get_chi_tilde(self): # if density basis overlap

        #drhom_norm = np.repeat(self.chi_monopole[self.layer_indices]
        #                       [:,:, 0, np.newaxis],
        #                       self.drho_array.shape[-1], axis=2)
        #chi_norm = self.chi_monopole[self.layer_indices]
        chi0_qwij = np.zeros([len(self.q_abs), len(self.frequencies),
                             self.dim, self.dim], dtype=complex)
        drho_array = self.drho_array.copy()# * drhom_norm
        for iw in range(len(self.frequencies)):
            drhom_norm = np.repeat(self.chi_monopole[self.layer_indices]
                                   [:,:, iw, np.newaxis],
                                   self.drho_array.shape[-1], axis=2)
            for iq in range(len(self.q_abs)):
                chi0_qwij[iq, iw] = np.dot(drho_array[:, iq] *
                                           drhom_norm[:, iq],
                                           self.basis_array.T * self.s[0]) \
                                           * self.dz
            
        return chi0_qwij
    """
        
    def get_Coulomb_Kernel(self, step_potential=False):
        kernel_qij = np.zeros([len(self.q_abs), self.dim, self.dim],
                              dtype=complex)
        factor = self.dim / self.n_layers
        for iq in range(len(self.q_abs)):
            if step_potential:  # Use step-function average over monopole layer
                kernel_qij[iq] = np.dot(self.basis_array[:, iq],
                                        self.dphi_array[:, iq].T) * self.dz
            else:  # Use induced potentials
                kernel_qij[iq] = np.dot(self.drho_array[:, iq],
                                        self.dphi_array[:, iq].T) * self.dz
                
        return kernel_qij

    def get_chi_matrix(self):

        """
        Dyson like equation;
        chi_full = chi_intra + chi_intra V_inter chi_full
        """
        Nls = self.n_layers
        q_abs = self.q_abs
        chi_m_iqw = self.chi_monopole
        chi_d_iqw = self.chi_dipole
        
        if self.kernel_qij is None:
            self.kernel_qij = self.get_Coulomb_Kernel()
        chi_qwij = np.zeros((len(self.q_abs), len(self.frequencies),
                                 self.dim, self.dim), dtype=complex)

        for iq in range(len(q_abs)):
            # Diagonal is set to zero
            kernel_ij = self.kernel_qij[iq].copy()
            np.fill_diagonal(kernel_ij, 0)
            for iw in range(0, len(self.frequencies)):
                chi_intra_i = chi_m_iqw[self.layer_indices, iq, iw]
                if self.chi_dipole is not None:
                    chi_intra_i = np.insert(chi_intra_i, np.arange(Nls) + 1,
                                            chi_d_iqw[self.layer_indices,
                                                      iq, iw])
                chi_intra_ij = np.diag(chi_intra_i)
                chi_qwij[iq, iw, :, :] = np.dot(np.linalg.inv(
                        np.eye(self.dim) - np.dot(chi_intra_ij, kernel_ij)),
                                                chi_intra_ij)
  
        return chi_qwij

    def get_eps_matrix(self, step_potential=False):
        Nls = self.n_layers
        self.kernel_qij =\
            self.get_Coulomb_Kernel(step_potential=step_potential)
        chi_qwij = self.get_chi_matrix()
        eps_qwij = np.zeros((len(self.q_abs), len(self.frequencies),
                             self.dim, self.dim), dtype=complex)

        for iq in range(len(self.q_abs)):
            kernel_ij = self.kernel_qij[iq]
            for iw in range(0, len(self.frequencies)):
                eps_qwij[iq, iw, :, :] = np.linalg.inv(\
                    np.eye(kernel_ij.shape[0]) + np.dot(kernel_ij,
                                                        chi_qwij[iq, iw,
                                                                 :, :]))
      
        return eps_qwij
    
    def get_exciton_screened_potential(self, e_distr, h_distr):
        v_screened_qw = np.zeros((len(self.q_abs),
                                  len(self.frequencies)))
        eps_qwij = self.get_eps_matrix(step_potential=True)
        h_distr = h_distr.transpose()
        kernel_qij = self.get_Coulomb_Kernel()

        for iq in range(0, len(self.q_abs)):
            ext_pot = np.dot(kernel_qij[iq], h_distr)
            for iw in range(0, len(self.frequencies)):
                v_screened_qw[iq, iw] =\
                    np.dot(e_distr,
                           np.dot(np.linalg.inv(eps_qwij[iq, iw, :, :]),
                                  ext_pot))
                        
        return -v_screened_qw

    def get_macroscopic_dielectric_constant(self, layers=None):
        constant_perturbation = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers
        
        if self.chi_dipole is not None:
            constant_perturbation = np.insert(constant_perturbation,
                                              np.arange(self.n_layers) + 1,
                                              np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)
        if layers is None:  # average over entire structure
            N = self.n_layers
            potential = constant_perturbation
        else:  # average over selected layers
            N = len(layers)
            potential = np.zeros([self.dim])
            index = layers * self.dim / self.n_layers
            potential[index] = 1.
        epsM_q = []
        eps_qij = self.get_eps_matrix(step_potential=True)[:, 0]
        for iq in range(len(self.q_abs)):
            eps_ij = eps_qij[iq]
            epsinv_ij = np.linalg.inv(eps_ij)
            epsinv_M = 1. / N * np.dot(np.array(potential) * layer_weight,
                                       np.dot(epsinv_ij,
                                              np.array(constant_perturbation)))
            epsM_q.append(1. / epsinv_M)
        return epsM_q

    def get_response(self, iw=0, dipole=False):
        """
        Induced density and potential due to constant perturbation
        """
        constant_perturbation = np.ones([self.n_layers])
        if self.chi_dipole is not None:
            constant_perturbation = np.insert(constant_perturbation,
                                              np.arange(self.n_layers) + 1,
                                              np.zeros([self.n_layers]))

        if dipole:
            constant_perturbation = self.z0 - self.z0[-1] / 2.
            constant_perturbation = np.insert(constant_perturbation,
                                              np.arange(self.n_layers) + 1,
                                              np.ones([self.n_layers]))
        print(constant_perturbation)
        chi_qij = self.get_chi_matrix()[:, iw]
        Vind_z = np.zeros((len(self.q_abs), len(self.z_big)))
        rhoind_z = np.zeros((len(self.q_abs), len(self.z_big)))
        
        drho_array = self.drho_array.copy()
        dphi_array = self.dphi_array.copy()
        
        for iq in range(len(self.q_abs)):
            chi_ij = chi_qij[iq]
            Vind_qi = np.dot(chi_ij, np.array(constant_perturbation))
            rhoind_z[iq] = np.dot(drho_array[:, iq].T, Vind_qi)
            Vind_z[iq] = np.dot(dphi_array[:, iq].T, Vind_qi)
        return self.z_big * Bohr, rhoind_z, Vind_z, self.z0 * Bohr
    
    def get_plasmon_eigenmodes(self):
        eps_qwij = self.get_eps_matrix()
        Nw = len(self.frequencies)
        Nq = len(self.q_abs)
        w_w = self.frequencies
        eig = np.zeros([Nq, Nw, self.dim], dtype=complex)
        vec = np.zeros([Nq, Nw, self.dim, self.dim],
                       dtype=complex)

        omega0 = [[] for i in range(Nq)]
        rho_z = [np.zeros([0, len(self.z_big)]) for i in range(Nq)]
        phi_z = [np.zeros([0, len(self.z_big)]) for i in range(Nq)]
        for iq in range(Nq):
            m = 0
            eig[iq, 0], vec[iq, 0] = np.linalg.eig(eps_qwij[iq, 0])
            vec_dual = np.linalg.inv(vec[iq, 0])
            for iw in range(1, Nw):
                eig[iq, iw], vec_p = np.linalg.eig(eps_qwij[iq, iw])
                vec_dual_p = np.linalg.inv(vec_p)
                overlap = np.abs(np.dot(vec_dual, vec_p))
                index = list(np.argsort(overlap)[:, -1])
                if len(np.unique(index)) < self.dim:  # add missing indices
                    addlist = []
                    removelist = []
                    for j in range(self.dim):
                        if index.count(j) < 1:
                            addlist.append(j)
                        if index.count(j) > 1:
                            for l in range(1, index.count(j)):
                                removelist.append(
                                    np.argwhere(np.array(index) == j)[l])
                    for j in range(len(addlist)):
                        index[removelist[j]] = addlist[j]
                vec[iq, iw] = vec_p[:, index]
                vec_dual = vec_dual_p[index, :]
                eig[iq, iw, :] = eig[iq, iw, index]
                klist = [k for k in range(self.dim) \
                         if (eig[iq, iw - 1, k] < 0 and eig[iq, iw, k] > 0)]
                for k in klist:  # Eigenvalue crossing
                    a = np.real((eig[iq, iw, k] - eig[iq, iw - 1, k]) / \
                                (w_w[iw] - w_w[iw - 1]))
                    # linear interp for crossing point
                    w0 = np.real(-eig[iq, iw - 1, k]) / a + w_w[iw - 1]
                    #eig0 = a * (w0 - w_w[iw-1]) + eig[iq, iw-1, k]
                    #print('crossing found at w = %1.2f eV'%w0)
                    rho = np.dot(self.drho_array[:, iq, :].T, vec_dual_p[k, :])
                    phi = np.dot(self.dphi_array[:, iq, :].T, vec_dual_p[k, :])
                    rho_z[iq] = np.append(rho_z[iq], rho[np.newaxis, :],
                                          axis=0)
                    phi_z[iq] = np.append(phi_z[iq], phi[np.newaxis, :],
                                          axis=0)
                    omega0[iq].append(w0)
                    m += 1

        return eig, self.z_big * Bohr, rho_z, phi_z, np.array(omega0)


"""TOOLS"""


def get_chi_2D(filenames, name=None):
    """Calculate the monopole and dipole contribution to the
    2D susceptibillity chi_2D, defined as

    ::

      \chi^M_2D(q, \omega) = \int\int dr dr' \chi(q, \omega, r,r') \\
                          = L \chi_{G=G'=0}(q, \omega)
      \chi^D_2D(q, \omega) = \int\int dr dr' z \chi(q, \omega, r,r') z'
                           = 1/L sum_{G_z,G_z'} z_factor(G_z)
                           chi_{G_z,G_z'} z_factor(G_z'),
      Where z_factor(G_z) =  +/- i e^{+/- i*G_z*z0}
      (L G_z cos(G_z L/2)-2 sin(G_z L/2))/G_z^2

    input parameters:
    
    filenames: list of str
        list of chi_wGG.pckl files for different q
    name: str
        name writing output files
    """
    nq = len(filenames)
    q_list_abs = []
    omega_w, pd, chi_wGG, q0 = read_chi_wGG(filenames[0])
    nw = omega_w.shape[0]
    r = pd.gd.get_grid_point_coordinates()
    z = r[2, 0, 0, :]
    L = pd.gd.cell_cv[2, 2]  # Length of cell in Bohr
    z0 = L / 2.  # position of layer
    chiM_2D_qw = np.zeros([nq, nw], dtype=complex)
    chiD_2D_qw = np.zeros([nq, nw], dtype=complex)
    drho_M_qz = np.zeros([nq, len(z)], dtype=complex)  # induced density
    drho_D_qz = np.zeros([nq, len(z)], dtype=complex)  # induced dipole density
    for iq in range(nq):
        if not iq == 0:
            omega_w, pd, chi_wGG, q0 = read_chi_wGG(filenames[iq])
        if q0 is not None:
            q = q0
        else:
            q = pd.K_qv
        npw = chi_wGG.shape[1]
        Gvec = pd.get_reciprocal_vectors(add_q=False)
        Glist = []
        for iG in range(npw):  # List of G with Gx,Gy = 0
            if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
                Glist.append(iG)
        chiM_2D_qw[iq] = L * chi_wGG[:, 0, 0]
        drho_M_qz[iq] += chi_wGG[0, 0, 0]
        q_abs = np.linalg.norm(q)
        q_list_abs.append(q_abs)
        for iG in Glist[1:]:
            G_z = Gvec[iG, 2]
            qGr_R = np.inner(G_z, z.T).T
            # Fourier transform to get induced density at \omega=0
            drho_M_qz[iq] += np.exp(1j * qGr_R) * chi_wGG[0, iG, 0]
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # integrate with z along both coordinates
                factor = z_factor(z0, L, G_z)
                factor1 = z_factor(z0, L, G_z1, sign=-1)
                chiD_2D_qw[iq, :] += 1. / L * factor * chi_wGG[:, iG, iG1] * \
                    factor1
                # induced dipole density due to V_ext = z
                drho_D_qz[iq, :] += 1. / L * np.exp(1j * qGr_R) * \
                    chi_wGG[0, iG, iG1] * factor1
    # Normalize induced densities with chi
    drho_M_qz /= np.repeat(chiM_2D_qw[:, 0, np.newaxis], drho_M_qz.shape[1],
                           axis=1)
    drho_D_qz /= np.repeat(chiD_2D_qw[:, 0, np.newaxis], drho_M_qz.shape[1],
                           axis=1)

    """ Returns q array, frequency array, chi2D monopole and dipole, induced
    densities and z array (all in Bohr)
    """
    pickle.dump((np.array(q_list_abs), omega_w, chiM_2D_qw, chiD_2D_qw, \
                     z, drho_M_qz, drho_D_qz), open(name + '-chi.pckl', 'w'))
    return np.array(q_list_abs) / Bohr, omega_w * Hartree, chiM_2D_qw, \
        chiD_2D_qw, z, drho_M_qz, drho_D_qz


def z_factor(z0, d, G, sign=1):
    factor = -1j * sign * np.exp(1j * sign * G * z0) * \
        (d * G * np.cos(G * d / 2.) - 2. * np.sin(G * d / 2.)) / G**2
    return factor


def z_factor2(z0, d, G, sign=1):
    factor = sign * np.exp(1j * sign * G * z0) * np.sin(G * d / 2.)
    return factor


def read_chi_wGG(name):
    """
    Read density response matrix calculated with the DielectricFunction
    module in GPAW.
    Returns frequency grid, gpaw.wavefunctions object, chi_wGG
    """
    fd = open(name)
    omega_w, pd, chi_wGG, q0, chi0_wvv = pickle.load(fd)
    nw = len(omega_w)
    nG = pd.ngmax
    chi_wGG = np.empty((nw, nG, nG), complex)
    for chi_GG in chi_wGG:
        chi_GG[:] = pickle.load(fd)
    return omega_w, pd, chi_wGG, q0


# Temporary, or should be rewritten!!!
def get_chiM_2D_from_old_DF(filenames_eps, read, qpoints, d=None,
                            write_chi0 = False, name = None):
    #rec_cell = reciprocal_cell*Bohr
    #q_points = np.loadtxt(filename_qpoints)
    #q_points = np.dot(q_points,rec_cell)
    #Gvec = pickle.load(open(filename_Gvec %0))
    #Gvec = np.dot(Gvec,rec_cell) # the cell has to be in bohr
    from gpaw.response.df0 import DF
    df = DF()
    df.read(read + str(qpoints[0]))
    cell = df.acell_cv
    Gvec = np.dot(df.Gvec_Gc,df.bcell_cv)
    nq = len(filenames_eps)#len(q_points[:,0])
    L = cell[2,2] # Length of cell in Bohr
    d /= Bohr # d in Bohr
    z0 = L/2. # position of layer
    npw = Gvec.shape[0]
    nw = df.Nw
    omega_w = df.w_w#[0.]
    q_points_abs = []
    Glist = []

    for iG in range(npw): # List of G with Gx,Gy = 0
        if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
            Glist.append(iG)
    epsM_2D_qw = np.zeros([nq, nw], dtype=complex)
    epsD_2D_qw = np.zeros([nq, nw], dtype=complex)
    chiM_2D_qw = np.zeros([nq, nw], dtype=complex)
    chiD_2D_qw = np.zeros([nq, nw], dtype=complex)
    VM_eff_qw = np.zeros([nq, nw], dtype=complex)
    for iq in range(nq):
        df.read(read + str(qpoints[iq]))
        la,la,la,eps_wGG, chi_wGG = pickle.load(open(filenames_eps[iq]))
        #chi_wGG = pickle.load(open(filenames_chi %iq))
        #chi_wGG = np.array(chi_wGG)
        eps_inv_wGG = np.zeros_like(eps_wGG, dtype = complex)
        for iw in range(nw):
            eps_inv_wGG[iw] = np.linalg.inv(eps_wGG[iw])
            eps_inv_wGG[iw] = np.identity(npw)
        del eps_wGG
        q = df.q_c#q_points[iq]
        q_abs = np.linalg.norm(q)
        q_points_abs.append(q_abs) # return q in Ang
        epsM_2D_inv = eps_inv_wGG[:, 0, 0]
        epsD_2D_inv = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = complex)
        chiM_2D = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = complex) #chi_wGG[:, 0, 0]#
        chiD_2D = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = complex)
        for iG in Glist[1:]:
            G_z = Gvec[iG, 2]
            epsM_2D_inv += 2./d * np.exp(1j*G_z*z0) * np.sin(G_z*d/2.) / G_z * eps_inv_wGG[:, iG, 0]
            
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # intregrate over entire cell for z and z'
                factor1 = z_factor(z0, L, G_z)
                factor2 = z_factor(z0, L, G_z1, sign=-1)
                chiD_2D += 1./L * factor1 * factor2 * chi_wGG[:, iG, iG1]
                # intregrate z over d for epsilon^-1
                #factor1 =  z_factor2(z0, d, G_z)
                #epsD_2D_inv += 2j / d / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #average
                #epsD_2D_inv += 1j * G_z * np.exp(1j*G_z*z0) * factor2 * eps_inv_wGG[:, iG, iG1]  #atz0
                factor1 =  z_factor(z0, d, G_z)
                epsD_2D_inv += 12. / d**3 / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #kristian
            
        epsM_2D_qw[iq, :] = 1. / epsM_2D_inv
        epsD_2D_qw[iq, :] = 1. / epsD_2D_inv
        chiM_2D_qw[iq, :] = L * chi_wGG[:, 0, 0] #chiM_2D#
        chiD_2D_qw[iq, :] = chiD_2D
        del chi_wGG,  eps_inv_wGG

    # Effective Coulomb interaction in 2D from eps_{2D}^{-1} = 1 + V_{eff} \chi_{2D}
    VM_eff_qw = (1. /epsM_2D_qw - 1) / chiM_2D_qw
    VD_eff_qw = (1. /epsD_2D_qw - 1) / chiD_2D_qw
    chi0M_2D_qw = (1 - epsM_2D_qw) * 1. / VM_eff_qw  # Chi0 from effective Coulomb
    chi0D_2D_qw = (1 - epsD_2D_qw) * 1. / VD_eff_qw
    pickle.dump((np.array(q_points_abs), omega_w, VM_eff_qw, VD_eff_qw,
                 chiM_2D_qw, chiD_2D_qw), open(name + '-chi.pckl', 'w'))
    pickle.dump((np.array(q_points_abs), omega_w, VM_eff_qw, VD_eff_qw,
                 chi0M_2D_qw, chi0D_2D_qw, chiM_2D_qw, chiD_2D_qw,
                 epsM_2D_qw, epsD_2D_qw), open(name + '-2D.pckl', 'w'))
        
    return np.array(q_points_abs), omega_w, chiM_2D_qw, chiD_2D_qw, VM_eff_qw, VD_eff_qw, epsM_2D_qw, epsD_2D_qw




