import numpy as np

from ase.units import Hartree, Bohr

from gpaw.io import Reader
from gpaw.io import Writer
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.utilities.tools import tri2full
from gpaw.tddft.units import eV_to_au


class KohnShamDecomposition(object):
    version = 1
    ulmtag = 'KSD'

    def __init__(self, paw=None, filename=None):
        self.filename = filename
        self.has_initialized = False
        self.world = paw.world
        self.log = paw.log
        self.wfs = paw.wfs

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
            assert S_MM.dtype == float
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
        NP = Nn * Nn
        P_p = np.array(P_p)

        dm_vMM = []
        for v in range(3):
            direction = np.array([0.]*3)
            direction[v] = 1.0
            magnitude = 1.0
            cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)
            kick_hamiltonian = KickHamiltonian(paw, cef)
            dm_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(
                kick_hamiltonian, paw.wfs, self.wfs.kpt_u[u], add_kinetic=False, root=-1)
            tri2full(dm_MM)  # TODO: do not use this
            dm_vMM.append(dm_MM)

        print 'Dipole moment matrix done'

        C0_nM = self.C0_unM[u]
        dm_vnn = []
        for v in range(3):
            dm_vnn.append(np.dot(C0_nM.conj(), np.dot(dm_vMM[v], C0_nM.T)))
        dm_vnn = np.array(dm_vnn)
        dm_vP = dm_vnn.reshape(3, -1)

        if 1:
            dm_vp = dm_vP[:,P_p]
        else:
            dm_pv = np.zeros((Np, 3)) * np.nan
            for p in range(Np):
                i = i_p[p]
                a = a_p[p]
                for v in range(3):
                    dm_pv[p,v] = dm_vnn[v][i,a]
        #print dm_pv

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

        for arg in ['fermilevel', 'only_ia', 'w_p', 'f_p', 'ia_p', 'P_p', 'dm_vp']:
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

        for arg in ['fermilevel', 'only_ia', 'w_p', 'f_p', 'ia_p', 'P_p', 'dm_vp']:
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
        rho_up = [rho_p]
        return rho_up

    def M_ia_from_M_p(self, M_p):
        i_p = self.ia_p[:,0]
        a_p = self.ia_p[:,1]
        imin = np.min(i_p)
        imax = np.max(i_p)
        amin = np.min(a_p)
        amax = np.max(a_p)

        M_ia = np.zeros((imax-imin+1, amax-amin+1), dtype=M_p.dtype)
        for M, i, a in zip(M_p, i_p, a_p):
            M_ia[i - imin, a - amin] = M
        return M_ia

    def plot_matrix(self, M_p):
        import matplotlib.pyplot as plt
        M_ia = self.M_ia_from_M_p(M_p)
        plt.imshow(M_ia, interpolation='none')
        plt.xlabel('a')
        plt.ylabel('i')

    def get_dipole_moment(self, rho_up):
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]
        dm_v = -np.dot(self.dm_vp, rho_p)
        if self.only_ia:
            dm_v *= 2
        return dm_v

    def get_contributions_table(self, rho_up, minweight=0.01):
        raise NotImplementedError()
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]

        # Weight
        weight_ip = []
        weightname_i = []
        weightname_i.append('% rho_p**2')
        weight_ip.append(rho_p**2 / np.sum(rho_p**2) * 100)
        #for v in range(3):
        #    weight_p = 2 * self.dm_vp[:,v] * rho_p / Hartree**2
        #    weightname_i.append('alpha_%s' % 'xyz'[v])
        #    weight_ip.append(weight_p)
        #    weightname_i.append('%% alpha_%s**2' % 'xyz'[v])
        #    weight_ip.append(weight_p**2 / np.sum(weight_p**2) * 100)

        weight_ip = np.array(weight_ip)

        Ni = weight_ip.shape[0]

        fmt_i = ['%14.12f'] * Ni
        p_s = np.argsort(np.absolute(weight_ip[0]))[::-1]
        restweight_i = np.zeros(Ni)
        totweight_i = np.zeros(Ni)

        txt = ''
        txt += ('# %6s %3s  %3s  %14s' + ' %14s' * Ni) % (
                ('p', 'i', 'a', 'Ediff (eV)') + tuple(weightname_i))
        for s, p in enumerate(p_s):
            i, a = self.ia_p[p]
            if weight_ip[0,p] > minweight:
                print ("  %6s %3d->%3d: %14.8f " + ' '.join(fmt_i)) % ((p, i,
                    a, self.w_p[p] * Hartree) + tuple(weight_ip[:,p]))
            else:
                restweight_i += weight_ip[:,p]
            totweight_i += weight_ip[:,p]
        print ("  %15s: %14s " + ' '.join(fmt_i)) % (('rest', '') + tuple(restweight_i))
        print ("  %15s: %14s " + ' '.join(fmt_i)) % (('total', '') + tuple(totweight_i))



