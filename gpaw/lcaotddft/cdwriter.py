import numpy as np

from gpaw.mpi import world
from ase.units import Hartree, alpha, Bohr
from gpaw.utilities.tools import coordinates
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.utilities import distribute_MM
from gpaw.utilities.blas import gemm, mmm


def skew(a):
    return 0.5 * (a - a.T)


def get_dX0(Ra_a, setups, partition):
    dX0_caii = []
    for _ in range(3):
        def shape(a):
            ni = setups[a].ni
            return ni, ni
        dX0_aii = partition.arraydict(shapes=shape, dtype=complex)
        for arr in dX0_aii.values():
            arr[:] = 0
        dX0_caii.append(dX0_aii)

    for a in partition.my_indices:
        Ra = Ra_a[a]

        rxnabla_iiv = setups[a].rxnabla_iiv.copy()
        nabla_iiv = setups[a].nabla_iiv.copy()

        for c in range(3):
            rxnabla_iiv[:,:,c]=skew(rxnabla_iiv[:,:,c])
            nabla_iiv[:,:,c]=skew(nabla_iiv[:,:,c])

        dX0_ii = -(Ra[1] * nabla_iiv[:, :, 2] - Ra[2] * nabla_iiv[:, :, 1] + rxnabla_iiv[:,:,0])
        dX1_ii = -(Ra[2] * nabla_iiv[:, :, 0] - Ra[0] * nabla_iiv[:, :, 2] + rxnabla_iiv[:,:,1])
        dX2_ii = -(Ra[0] * nabla_iiv[:, :, 1] - Ra[1] * nabla_iiv[:, :, 0] + rxnabla_iiv[:,:,2])

        for c, dX_ii in enumerate([dX0_ii, dX1_ii, dX2_ii]):
            assert not dX0_caii[c][a].any()
            dX0_caii[c][a][:] = dX_ii

    return dX0_caii


def calculate_E(dX0_caii, kpt_u, bfs, correction, r_cG):
    Mstart = correction.Mstart
    Mstop = correction.Mstop
    mynao = Mstop - Mstart
    nao = bfs.Mmax

    assert bfs.Mstart == Mstart
    assert bfs.Mstop == Mstop

    E_cmM = np.zeros((3, mynao, nao), dtype=complex)
    A_cmM = np.zeros((3, mynao, nao), dtype=complex)

    for c in range(3):
        for kpt in kpt_u:
            assert kpt.k == 0
            correction.calculate(kpt.q, dX0_caii[c], E_cmM[c],
                                 Mstart, Mstop)

    bfs.calculate_potential_matrix_derivative(r_cG[0], A_cmM, 0)
    E_cmM[1]-=A_cmM[2]
    E_cmM[2]+=A_cmM[1]

    A_cmM[:]=0.0
    bfs.calculate_potential_matrix_derivative(r_cG[1], A_cmM, 0)
    E_cmM[0]+=A_cmM[2]
    E_cmM[2]-=A_cmM[0]

    A_cmM[:]=0.0
    bfs.calculate_potential_matrix_derivative(r_cG[2], A_cmM, 0)
    E_cmM[0]-=A_cmM[1]
    E_cmM[1]+=A_cmM[0]
    return E_cmM


def calculate_cd_from_rho_and_e(rho_xx, E_cxx):
    # (Can save time by doing imag/real algebra explicitly, but this
    #  probably doesn't matter.)
    return -(rho_xx[None] * E_cxx).sum(axis=-1).sum(axis=-1).imag


class BlacsEMatrix:
    def __init__(self, ksl, E_cmm):
        self.ksl = ksl
        self.E_cmm = E_cmm

    @classmethod
    def redist_from_raw(cls, ksl, E_cmM):
        assert ksl.using_blacs
        E_cmm = ksl.distribute_overlap_matrix(E_cmM)
        return BlacsEMatrix(ksl, E_cmm)

    def calculate_cd(self, rho_mm):
        print(rho_mm.shape, self.E_cmm.shape)
        cd_c = calculate_cd_from_rho_and_e(rho_mm, self.E_cmm)
        self.ksl.mmdescriptor.blacsgrid.comm.sum(cd_c)
        return cd_c


class SerialEMatrix:
    def __init__(self, ksl, E_cMM):
        self.ksl = ksl
        self.E_cMM = E_cMM

    def calculate_cd(self, rho_MM):
        return calculate_cd_from_rho_and_e(rho_MM, self.E_cMM)

def debug_msg(msg):
    print('[%01d/%01d]: %s' % (world.rank, world.size, msg))

def convert_repr(r):
    # Integer
    try:
        return int(r)
    except ValueError:
        pass
    # Boolean
    b = {repr(False): False, repr(True): True}.get(r, None)
    if b is not None:
        return b
    # String
    s = r[1:-1]
    if repr(s) == r:
        return s
    raise RuntimeError('Unknown value: %s' % r)

class CDWriter(TDDFTObserver):
    version=1

    def __init__(self, paw, filename, center=False, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
        self.dmat = DensityMatrix(paw)  # XXX
        self.ksl = paw.wfs.ksl
        if paw.niter == 0:
            # Initialize
            self.do_center = center
            if self.master:
                self.fd = open(filename, 'w')
        else:
            # Read and continue
            self.read_header(filename)
            if self.master:
                self.fd = open(filename, 'a')

        gd = paw.wfs.gd
        self.timer=paw.timer

        R0 = 0.5 * np.diag(gd.cell_cv)
        Ra_a = paw.atoms.positions / Bohr - R0[None, :]
        r_cG, r2_G = coordinates(gd, origin=R0)

        dX0_caii = get_dX0(Ra_a, paw.setups, paw.hamiltonian.dH_asp.partition)

        E_cmM = calculate_E(dX0_caii, paw.wfs.kpt_u,
                            paw.wfs.basis_functions,
                            paw.wfs.atomic_correction, r_cG)

        if self.ksl.using_blacs:
            self.e_matrix = BlacsEMatrix.redist_from_raw(self.ksl, E_cmM)
        else:
            gd.comm.sum(E_cmM)
            self.e_matrix = SerialEMatrix(self.ksl, E_cmM)

    def _write(self, line):
        if self.master:
            self.fd.write(line)
            self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = '# %s[version=%s]' % (self.__class__.__name__, self.version)
        line += ('(center=%s)\n' %
                 (repr(self.do_center)))
        line += ('# %15s %22s %22s %22s\n' %
                 ('time', 'cmx', 'cmy', 'cmz'))
        self._write(line)

    def read_header(self, filename):
        with open(filename, 'r') as f:
            line = f.readline()
        m_i = re.split("[^a-zA-Z0-9_=']+", line[2:])
        name = m_i.pop(0)
        assert name == self.__class__.__name__
        for m in m_i:
            if '=' not in m:
                continue
            k, v = m.split('=')
            v = convert_repr(v)
            if k == 'version':
                assert v == self.version
                continue
            # Translate key
            k = {'center': 'do_center'}[k]
            setattr(self, k, v)

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def calculate_cd_moment(self, paw, center=True):
        u = 0
        kpt = paw.wfs.kpt_u[0]
        ksl = self.e_matrix.ksl
        rho_mm = ksl.calculate_blocked_density_matrix(kpt.f_n, kpt.C_nM)
        cd_c = self.e_matrix.calculate_cd(rho_mm)
        assert cd_c.shape == (3,)
        assert cd_c.dtype == float
        return cd_c

    def _write_cd(self, paw):
        time = paw.time
        cd = self.calculate_cd_moment(paw,center=self.do_center)
        print('cd', cd)
        line = ('%20.8lf %22.12le %22.12le %22.12le\n' %
          (time, cd[0], cd[1], cd[2]))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_cd(paw)

    def __del__(self):
        if self.master:
            self.fd.close()
        TDDFTObserver.__del__(self)

