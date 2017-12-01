import numpy as np

from ase.units import Hartree, Bohr

from gpaw.io import Reader
from gpaw.io import Writer
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.utilities import pack
from gpaw.utilities.tools import tri2full


class KohnShamDecomposition(object):
    version = 1
    ulmtag = 'KSD'
    readwrite_attrs = ['fermilevel', 'only_ia', 'w_p', 'f_p', 'ia_p',
                       'P_p', 'dm_vp']

    def __init__(self, paw=None, filename=None):
        self.filename = filename
        self.has_initialized = False
        self.world = paw.world
        self.log = paw.log
        self.wfs = paw.wfs
        self.density = paw.density

        if self.wfs.bd.comm.size > 1:
            raise RuntimeError('Band parallelization is not supported')
        if len(self.wfs.kpt_u) > 1:
            raise RuntimeError('K-points are not supported')

        if filename is not None:
            self.read(filename)
            return

    def initialize(self, paw, min_occdiff=1e-3, only_ia=True):
        if self.has_initialized:
            return
        paw.initialize_positions()
        # paw.set_positions()

        if self.wfs.gd.pbc_c.any():
            self.C0_dtype = complex
        else:
            self.C0_dtype = float

        # Take quantities
        self.fermilevel = paw.occupations.get_fermi_level()
        self.S_uMM = []
        self.C0_unM = []
        self.eig_un = []
        self.occ_un = []
        for kpt in self.wfs.kpt_u:
            S_MM = kpt.S_MM
            assert np.max(np.absolute(S_MM.imag)) == 0.0
            S_MM = S_MM.real
            self.S_uMM.append(S_MM)

            C_nM = kpt.C_nM
            if self.C0_dtype == float:
                assert np.max(np.absolute(C_nM.imag)) == 0.0
                C_nM = C_nM.real
            self.C0_unM.append(C_nM)

            self.eig_un.append(kpt.eps_n)
            self.occ_un.append(kpt.f_n)

        # TODO: do the rest of the function with K-points

        # Construct p = (i, a) pairs
        u = 0
        Nn = self.wfs.bd.nbands
        eig_n = self.eig_un[u]
        occ_n = self.occ_un[u]

        self.only_ia = only_ia
        f_p = []
        w_p = []
        i_p = []
        a_p = []
        ia_p = []
        i0 = 0
        for i in range(i0, Nn):
            if only_ia:
                a0 = i + 1
            else:
                a0 = 0
            for a in range(a0, Nn):
                f = occ_n[i] - occ_n[a]
                if only_ia and f < min_occdiff:
                    continue
                w = eig_n[a] - eig_n[i]
                f_p.append(f)
                w_p.append(w)
                i_p.append(i)
                a_p.append(a)
                ia_p.append((i, a))
        f_p = np.array(f_p)
        w_p = np.array(w_p)
        i_p = np.array(i_p, dtype=int)
        a_p = np.array(a_p, dtype=int)
        ia_p = np.array(ia_p, dtype=int)

        # Sort according to energy difference
        p_s = np.argsort(w_p)
        f_p = f_p[p_s]
        w_p = w_p[p_s]
        i_p = i_p[p_s]
        a_p = a_p[p_s]
        ia_p = ia_p[p_s]

        Np = len(f_p)
        P_p = []
        for p in range(Np):
            P = np.ravel_multi_index(ia_p[p], (Nn, Nn))
            P_p.append(P)
        P_p = np.array(P_p)

        dm_vMM = []
        for v in range(3):
            direction = np.zeros(3, dtype=float)
            direction[v] = 1.0
            magnitude = 1.0
            cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)
            kick_hamiltonian = KickHamiltonian(paw, cef)
            dm_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(
                kick_hamiltonian, paw.wfs, self.wfs.kpt_u[u],
                add_kinetic=False, root=-1)
            tri2full(dm_MM)  # TODO: do not use this
            dm_vMM.append(dm_MM)

        print 'Dipole moment matrix done'

        C0_nM = self.C0_unM[u]
        dm_vnn = []
        for v in range(3):
            dm_vnn.append(np.dot(C0_nM.conj(), np.dot(dm_vMM[v], C0_nM.T)))
        dm_vnn = np.array(dm_vnn)
        dm_vP = dm_vnn.reshape(3, -1)

        dm_vp = dm_vP[:, P_p]

        self.w_p = w_p
        self.f_p = f_p
        self.ia_p = ia_p
        self.P_p = P_p
        self.dm_vp = dm_vp

        self.has_initialized = True

    def write(self, filename):
        self.log('%s: Writing to %s' % (self.__class__.__name__, filename))
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        writer.write(version=self.__class__.version)

        wfs = self.wfs
        writer.write(ha=Hartree)
        write_uMM(wfs, writer, 'S_uMM', self.S_uMM)
        wfs.write_wave_functions(writer)
        wfs.write_eigenvalues(writer)
        wfs.write_occupations(writer)
        # write_unM(wfs, writer, 'C0_unM', self.C0_unM)
        # write_un(wfs, writer, 'eig_un', self.eig_un)
        # write_un(wfs, writer, 'occ_un', self.occ_un)

        for arg in self.readwrite_attrs:
            writer.write(arg, getattr(self, arg))

        writer.close()

    def read(self, filename):
        reader = Reader(filename)
        tag = reader.get_tag()
        if tag != self.__class__.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        version = reader.version
        if version != self.__class__.version:
            raise RuntimeError('Unknown version %s' % version)

        wfs = self.wfs
        self.S_uMM = read_uMM(wfs, reader, 'S_uMM')
        wfs.read_wave_functions(reader)
        wfs.read_eigenvalues(reader)
        wfs.read_occupations(reader)

        self.C0_unM = []
        self.eig_un = []
        self.occ_un = []
        for kpt in self.wfs.kpt_u:
            C_nM = kpt.C_nM
            self.C0_unM.append(C_nM)
            self.eig_un.append(kpt.eps_n)
            self.occ_un.append(kpt.f_n)

        for arg in self.readwrite_attrs:
            setattr(self, arg, getattr(reader, arg))

        reader.close()
        self.has_initialized = True

    def transform(self, rho_uMM):
        assert len(rho_uMM) == 1, 'K-points not implemented'
        u = 0
        C0_nM = self.C0_unM[u]
        S_MM = self.S_uMM[u]
        rho_MM = rho_uMM[u]
        # KS decomposition
        C0S_nM = np.dot(C0_nM, S_MM)
        rho_nn = np.dot(np.dot(C0S_nM, rho_MM), C0S_nM.T.conjugate())
        rho_P = rho_nn.reshape(-1)

        # Remove de-excitation terms
        rho_p = rho_P[self.P_p]
        if self.only_ia:
            rho_p *= 2

        rho_up = [rho_p]
        return rho_up

    def ialims(self):
        i_p = self.ia_p[:, 0]
        a_p = self.ia_p[:, 1]
        imin = np.min(i_p)
        imax = np.max(i_p)
        amin = np.min(a_p)
        amax = np.max(a_p)
        return imin, imax, amin, amax

    def M_p_to_M_ia(self, M_p):
        return self.M_ia_from_M_p(M_p)

    def M_ia_from_M_p(self, M_p):
        imin, imax, amin, amax = self.ialims()
        M_ia = np.zeros((imax - imin + 1, amax - amin + 1), dtype=M_p.dtype)
        for M, (i, a) in zip(M_p, self.ia_p):
            M_ia[i - imin, a - amin] = M
        return M_ia

    def plot_matrix(self, M_p):
        import matplotlib.pyplot as plt
        M_ia = self.M_ia_from_M_p(M_p)
        plt.imshow(M_ia, interpolation='none')
        plt.xlabel('a')
        plt.ylabel('i')

    def get_dipole_moment_contributions(self, rho_up):
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]
        dmrho_vp = - self.dm_vp * rho_p
        return dmrho_vp

    def get_dipole_moment(self, rho_up):
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]
        dm_v = - np.dot(self.dm_vp, rho_p)
        return dm_v

    def get_density(self, rho_up, density='comp'):
        density_type = density
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        kpt = self.wfs.kpt_u[u]
        rho_p = rho_up[u]
        C0_nM = self.C0_unM[u]

        rho_ia = self.M_ia_from_M_p(rho_p)
        imin, imax, amin, amax = self.ialims()
        C0_iM = C0_nM[imin:(imax + 1)]
        C0_aM = C0_nM[amin:(amax + 1)]

        rho_MM = np.dot(C0_iM.T, np.dot(rho_ia, C0_aM.conj()))
        rho_MM = 0.5 * (rho_MM + rho_MM.T)

        rho_G = self.density.gd.zeros()
        assert kpt.q == 0
        rho_MM = rho_MM.astype(self.wfs.dtype)
        self.wfs.basis_functions.construct_density(rho_MM, rho_G, kpt.q)

        # Uncomment this if you want to add the static part
        # rho_G += self.density.nct_G

        if density_type == 'pseudocoarse':
            return rho_G

        rho_g = self.density.finegd.zeros()
        self.density.distribute_and_interpolate(rho_G, rho_g)
        rho_G = None

        if density_type == 'pseudo':
            return rho_g

        if density_type == 'comp':
            D_asp = self.density.atom_partition.arraydict(
                self.density.D_asp.shapes_a)
            Q_aL = {}
            for a, D_sp in D_asp.items():
                P_Mi = self.wfs.P_aqMi[a][kpt.q]
                assert np.max(np.absolute(P_Mi.imag)) == 0
                P_Mi = P_Mi.real
                assert P_Mi.dtype == float
                D_ii = np.dot(np.dot(P_Mi.T.conj(), rho_MM), P_Mi)
                D_sp[:] = pack(D_ii)[np.newaxis, :]
                Q_aL[a] = np.dot(D_sp.sum(axis=0),
                                 self.wfs.setups[a].Delta_pL)
            tmp_g = self.density.finegd.zeros()
            self.density.ghat.add(tmp_g, Q_aL)
            rho_g += tmp_g
            return rho_g

        raise RuntimeError('Unknown density type: %s' % density_type)

    def get_contributions_table(self, weight_p, minweight=0.01,
                                zero_fermilevel=True):
        assert weight_p.dtype == float
        u = 0  # TODO

        absweight_p = np.absolute(weight_p)
        tot_weight = weight_p.sum()
        rest_weight = tot_weight
        eig_n = self.eig_un[u]
        if zero_fermilevel:
            eig_n -= self.fermilevel

        txt = ''
        txt += ('# %6s %3s(%8s)  %3s(%8s)  %12s %14s\n' %
                ('p', 'i', 'eV', 'a', 'eV', 'Ediff (eV)', 'weight'))
        p_s = np.argsort(absweight_p)[::-1]
        for s, p in enumerate(p_s):
            i, a = self.ia_p[p]
            if absweight_p[p] < minweight:
                break
            txt += ('  %6s %3d(%8.3f)->%3d(%8.3f): %12.4f %+14.4f\n' %
                    (p, i, eig_n[i] * Hartree, a, eig_n[a] * Hartree,
                     self.w_p[p] * Hartree, weight_p[p]))
            rest_weight -= weight_p[p]
        txt += ('  %15s: %14s %+14.8f\n' %
                ('rest', '', rest_weight))
        txt += ('  %15s: %14s %+14.8f\n' %
                ('total', '', tot_weight))
        return txt
