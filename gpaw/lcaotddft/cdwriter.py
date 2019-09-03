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
    nao = bfs.Mmax
    E_cMM = np.zeros((3, nao, nao), dtype=complex)
    A_cMM = np.zeros((3, nao, nao), dtype=complex)

    for c in range(3):
        for kpt in kpt_u:
            assert kpt.k == 0
            correction.calculate(kpt.q, dX0_caii[c], E_cMM[c],
                                 correction.Mstart, correction.Mstop)

    bfs.calculate_potential_matrix_derivative(r_cG[0], A_cMM, 0)
    E_cMM[1]-=A_cMM[2]
    E_cMM[2]+=A_cMM[1]
    A_cMM[:]=0.0

    bfs.calculate_potential_matrix_derivative(r_cG[1], A_cMM, 0)
    E_cMM[0]+=A_cMM[2]
    E_cMM[2]-=A_cMM[0]
    A_cMM[:]=0.0

    bfs.calculate_potential_matrix_derivative(r_cG[2], A_cMM, 0)
    E_cMM[0]-=A_cMM[1]
    E_cMM[1]+=A_cMM[0]
    return E_cMM


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

        E_cMM = calculate_E(dX0_caii, paw.wfs.kpt_u,
                            paw.wfs.basis_functions,
                            paw.wfs.atomic_correction, r_cG)
        gd.comm.sum(E_cMM)

        if self.ksl.using_blacs:
            E_cmm = []
            for c in range(3):
                E_mm = distribute_MM(paw.wfs, E_cMM[c])
                E_cmm.append(E_mm)
            E_cMM = np.array(E_cmm)
        self.E_cMM = E_cMM

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
        assert m_i.pop(0) == self.__class__.__name__
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
        rho_MM = self.dmat.get_density_matrix()[u]
        debug_msg('rho %s' % str(rho_MM.shape))
        debug_msg('E %s' % str(self.E_cMM[0].shape))

        cd_c = []
        for c in range(3):
            if self.ksl.using_blacs:
                cd = self.ksl.block_comm.sum(np.sum(rho_MM * 1j * self.E_cMM[c]))
            else:
                cd = np.sum(rho_MM * 1j * self.E_cMM[c])
            cd_c.append(cd)
        return np.array(cd_c).real


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

