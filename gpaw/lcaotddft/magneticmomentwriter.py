import re
import numpy as np

from ase.units import Bohr
from gpaw.fd_operators import Gradient
from gpaw.utilities.tools import coordinates
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.dipolemomentwriter import convert_repr


def skew(a):
    return 0.5 * (a - a.T)


def calculate_mm_on_grid(wfs, grad_v, r_cG, dX0_caii, timer,
                         only_pseudo=False):
    gd = wfs.gd
    mode = wfs.mode
    bd = wfs.bd
    kpt_u = wfs.kpt_u

    timer.start('Magnetic moment')

    grad_psit_vG = gd.empty(3, dtype=complex)
    pseudo_rxnabla_v = np.zeros(3, dtype=complex)
    paw_rxnabla_v = np.zeros(3, dtype=complex)
    if mode == 'lcao':
        psit_G = gd.empty(dtype=complex)

    for kpt in kpt_u:
        for n, f in enumerate(kpt.f_n):
            if mode == 'lcao':
                psit_G[:] = 0.0
                wfs.basis_functions.lcao_to_grid(kpt.C_nM[n], psit_G, kpt.q)
            else:
                psit_G = kpt.psit_nG[n]

            for v in range(3):
                grad_v[v].apply(psit_G, grad_psit_vG[v], kpt.phase_cd)

            timer.start('Pseudo')

            def rxnabla(v1, v2):
                return f * gd.integrate(psit_G.conjugate() *
                                        (r_cG[v1] * grad_psit_vG[v2] -
                                         r_cG[v2] * grad_psit_vG[v1]))

            # rxnabla   = <psi1| r x nabla |psi2>
            # rxnabla_x = <psi1| r_y nabla_z - r_z nabla_y |psi2>
            # rxnabla_y = <psi1| r_z nabla_x - r_x nabla_z |psi2>
            # rxnabla_z = <psi1| r_x nabla_y - r_y nabla_x |psi2>
            pseudo_rxnabla_v[0] += rxnabla(1, 2)
            pseudo_rxnabla_v[1] += rxnabla(2, 0)
            pseudo_rxnabla_v[2] += rxnabla(0, 1)
            timer.stop('Pseudo')

            if not only_pseudo:
                timer.start('PAW')
                for a, P_ni in kpt.P_ani.items():
                    P_i = P_ni[n]
                    for v in range(3):
                        PdXP = np.dot(P_i.conj(), np.dot(dX0_caii[v][a], P_i))
                        paw_rxnabla_v[v] += f * PdXP
                timer.stop('PAW')

    bd.comm.sum(paw_rxnabla_v)
    gd.comm.sum(paw_rxnabla_v)

    bd.comm.sum(pseudo_rxnabla_v)

    rxnabla_v = pseudo_rxnabla_v + paw_rxnabla_v

    timer.stop('Magnetic moment')

    return rxnabla_v.imag


def get_dX0(Ra_a, setups, partition):
    # augmentation contributions to magnetic moment
    # <psi1| r x nabla |psi2> = <psi1| (r - Ra + Ra) x nabla |psi2>
    #                         = <psi1| (r - Ra) x nabla |psi2>
    #                             + Ra x <psi1| nabla |psi2>

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
            rxnabla_iiv[:, :, c] = skew(rxnabla_iiv[:, :, c])
            nabla_iiv[:, :, c] = skew(nabla_iiv[:, :, c])

        def Rxnabla(v1, v2):
            return (Ra[v1] * nabla_iiv[:, :, v2] -
                    Ra[v2] * nabla_iiv[:, :, v1])

        # rxnabla: <psi1| (r - Ra) x nabla |psi2>
        # Rxnabla: Ra x <psi1| nabla |psi2>
        # Rxnabla_x = (Ra_y nabla_z - Ra_z nabla_y)
        # Rxnabla_y = (Ra_z nabla_x - Ra_x nabla_z)
        # Rxnabla_z = (Ra_x nabla_y - Ra_y nabla_x)
        dX0_ii = Rxnabla(1, 2) + rxnabla_iiv[:, :, 0]
        dX1_ii = Rxnabla(2, 0) + rxnabla_iiv[:, :, 1]
        dX2_ii = Rxnabla(0, 1) + rxnabla_iiv[:, :, 2]

        for c, dX_ii in enumerate([dX0_ii, dX1_ii, dX2_ii]):
            assert not dX0_caii[c][a].any()
            dX0_caii[c][a][:] = dX_ii

    return dX0_caii


def calculate_E(dX0_caii, kpt_u, bfs, correction, r_cG, only_pseudo=False):
    Mstart = correction.Mstart
    Mstop = correction.Mstop
    mynao = Mstop - Mstart
    nao = bfs.Mmax

    assert bfs.Mstart == Mstart
    assert bfs.Mstop == Mstop

    E_cmM = np.zeros((3, mynao, nao), dtype=complex)
    A_cmM = np.zeros((3, mynao, nao), dtype=complex)

    if not only_pseudo:
        for c in range(3):
            for kpt in kpt_u:
                assert kpt.k == 0
                correction.calculate(kpt.q, dX0_caii[c], E_cmM[c],
                                     Mstart, Mstop)
        E_cmM *= -1

    bfs.calculate_potential_matrix_derivative(r_cG[0], A_cmM, 0)
    E_cmM[1] -= A_cmM[2]
    E_cmM[2] += A_cmM[1]

    A_cmM[:] = 0.0
    bfs.calculate_potential_matrix_derivative(r_cG[1], A_cmM, 0)
    E_cmM[0] += A_cmM[2]
    E_cmM[2] -= A_cmM[0]

    A_cmM[:] = 0.0
    bfs.calculate_potential_matrix_derivative(r_cG[2], A_cmM, 0)
    E_cmM[0] -= A_cmM[1]
    E_cmM[1] += A_cmM[0]

    # The matrix should be real
    assert np.max(np.absolute(E_cmM.imag)) == 0.0
    E_cmM = E_cmM.real.copy()
    return E_cmM


def calculate_mm_from_rho_and_e(rho_xx, E_cxx):
    assert E_cxx.dtype == float
    return -np.sum(rho_xx.imag * E_cxx, axis=(1, 2))


class BlacsEMatrix:
    def __init__(self, ksl, E_cmm):
        self.ksl = ksl
        self.E_cmm = E_cmm

    @classmethod
    def redist_from_raw(cls, ksl, E_cmM):
        assert ksl.using_blacs
        E_cmm = ksl.distribute_overlap_matrix(E_cmM)
        return BlacsEMatrix(ksl, E_cmm)

    def calculate_mm(self, rho_mm):
        mm_c = calculate_mm_from_rho_and_e(rho_mm, self.E_cmm)
        self.ksl.mmdescriptor.blacsgrid.comm.sum(mm_c)
        return mm_c


class SerialEMatrix:
    def __init__(self, ksl, E_cMM):
        self.ksl = ksl
        self.E_cMM = E_cMM

    def calculate_mm(self, rho_MM):
        return calculate_mm_from_rho_and_e(rho_MM, self.E_cMM)


class MagneticMomentWriter(TDDFTObserver):
    version = 1

    def __init__(self, paw, filename, center=True, interval=1,
                 calculate_on_grid=False, only_pseudo=False):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
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

        mode = paw.wfs.mode
        assert mode in ['fd', 'lcao'], 'unknown mode: {}'.format(mode)
        if mode == 'fd':
            self.calculate_on_grid = True
        else:
            self.calculate_on_grid = calculate_on_grid

        gd = paw.wfs.gd
        self.timer = paw.timer

        assert center
        # TODO: change R0 to choose another origin
        R0 = 0.5 * np.diag(gd.cell_cv)
        Ra_a = paw.atoms.positions / Bohr - R0[None, :]
        r_cG, _ = coordinates(gd, origin=R0)

        dX0_caii = get_dX0(Ra_a, paw.setups, paw.hamiltonian.dH_asp.partition)

        if self.calculate_on_grid:
            self.only_pseudo = only_pseudo
            self.r_cG = r_cG
            self.dX0_caii = dX0_caii

            grad_v = []
            for v in range(3):
                grad_v.append(Gradient(gd, v, dtype=complex, n=2))
            self.grad_v = grad_v
        else:
            E_cmM = calculate_E(dX0_caii, paw.wfs.kpt_u,
                                paw.wfs.basis_functions,
                                paw.wfs.atomic_correction, r_cG,
                                only_pseudo=only_pseudo)

            self.dmat = DensityMatrix(paw)  # XXX
            ksl = paw.wfs.ksl
            if ksl.using_blacs:
                self.e_matrix = BlacsEMatrix.redist_from_raw(ksl, E_cmM)
            else:
                gd.comm.sum(E_cmM)
                self.e_matrix = SerialEMatrix(ksl, E_cmM)

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

    def calculate_mm(self, paw):
        if self.calculate_on_grid:
            mm_c = calculate_mm_on_grid(paw.wfs, self.grad_v, self.r_cG,
                                        self.dX0_caii, self.timer,
                                        only_pseudo=self.only_pseudo)
        else:
            u = 0
            kpt = paw.wfs.kpt_u[u]
            ksl = self.e_matrix.ksl
            rho_mm = ksl.calculate_blocked_density_matrix(kpt.f_n, kpt.C_nM)
            mm_c = self.e_matrix.calculate_mm(rho_mm)
        assert mm_c.shape == (3,)
        assert mm_c.dtype == float
        return mm_c

    def _write_mm(self, paw):
        time = paw.time
        mm = self.calculate_mm(paw)
        line = ('%20.8lf %22.12le %22.12le %22.12le\n' %
                (time, mm[0], mm[1], mm[2]))
        self._write(line)

    def _update(self, paw):
        if hasattr(paw, 'action'):
            if paw.action == 'init':
                self._write_header(paw)
            elif paw.action == 'kick':
                self._write_kick(paw)
        self._write_mm(paw)

    def __del__(self):
        if self.master:
            self.fd.close()
        TDDFTObserver.__del__(self)
