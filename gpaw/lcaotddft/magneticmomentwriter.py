import re
import numpy as np

from ase.units import Bohr
from gpaw.fd_operators import Gradient
from gpaw.typing import ArrayLike
from gpaw.utilities.tools import coordinates
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.densitymatrix import DensityMatrix


def calculate_magnetic_moment_on_grid(wfs, grad_v, r_cG, dM_caii,
                                      timer, *,
                                      only_pseudo=False):
    """Calculate magnetic moment on grid.

    Parameters
    ----------
    wfs
        Wave functions object
    grad_v
        List of gradient operators
    r_cG
        Grid point coordinates
    dM_caii
        Atomic PAW corrections for magnetic moment
    timer
        Timer object
    only_pseudo
        If true, do not add atomic corrections

    Returns
    -------
    Magnetic moment vector.
    """
    gd = wfs.gd
    mode = wfs.mode
    bd = wfs.bd
    kpt_u = wfs.kpt_u

    timer.start('Magnetic moment')

    rxnabla_v = np.zeros(3, dtype=complex)

    timer.start('Pseudo')
    if mode == 'lcao':
        psit_G = gd.empty(dtype=complex)
    nabla_psit_vG = gd.empty(3, dtype=complex)
    for kpt in kpt_u:
        for n, f in enumerate(kpt.f_n):
            if mode == 'lcao':
                psit_G[:] = 0.0
                wfs.basis_functions.lcao_to_grid(kpt.C_nM[n], psit_G, kpt.q)
            else:
                psit_G = kpt.psit_nG[n]

            for v in range(3):
                grad_v[v].apply(psit_G, nabla_psit_vG[v], kpt.phase_cd)

            # rxnabla   = <psi1| r x nabla |psi2>
            # rxnabla_x = <psi1| r_y nabla_z - r_z nabla_y |psi2>
            # rxnabla_y = <psi1| r_z nabla_x - r_x nabla_z |psi2>
            # rxnabla_z = <psi1| r_x nabla_y - r_y nabla_x |psi2>
            for v in range(3):
                v1 = (v + 1) % 3
                v2 = (v + 2) % 3
                rnabla_psit_G = (r_cG[v1] * nabla_psit_vG[v2]
                                 - r_cG[v2] * nabla_psit_vG[v1])
                rxnabla_v[v] += f * gd.integrate(psit_G.conj() * rnabla_psit_G)

    timer.stop('Pseudo')

    if not only_pseudo:
        timer.start('PAW')
        paw_rxnabla_v = np.zeros(3, dtype=complex)
        for kpt in kpt_u:
            for v in range(3):
                for a, P_ni in kpt.P_ani.items():
                    paw_rxnabla_v[v] += np.einsum('n,ni,ij,nj',
                                                  kpt.f_n, P_ni.conj(),
                                                  dM_caii[v][a], P_ni,
                                                  optimize=True)
        gd.comm.sum(paw_rxnabla_v)
        rxnabla_v += paw_rxnabla_v
        timer.stop('PAW')

    bd.comm.sum(rxnabla_v)
    timer.stop('Magnetic moment')

    return -0.5 * rxnabla_v.imag


def get_magnetic_moment_atomic_corrections(R_av, setups, partition):
    """Calculate atomic PAW augmentation corrections for magnetic moment.

    Parameters
    ----------
    R_av
        Atom positions
    setups
        PAW setups object
    partition
        Partition object

    Returns
    -------
    Atomic correction matrices.
    """
    # augmentation contributions to magnetic moment
    # <psi1| r x nabla |psi2> = <psi1| (r - Ra + Ra) x nabla |psi2>
    #                         = <psi1| (r - Ra) x nabla |psi2>
    #                             + Ra x <psi1| nabla |psi2>

    def shape(a):
        ni = setups[a].ni
        return ni, ni

    dM_caii = []
    for _ in range(3):
        dM_aii = partition.arraydict(shapes=shape, dtype=complex)
        dM_caii.append(dM_aii)

    for a in partition.my_indices:
        Ra_v = R_av[a]
        rxnabla_iiv = setups[a].rxnabla_iiv
        nabla_iiv = setups[a].nabla_iiv

        # rxnabla   = <psi1| (r - Ra) x nabla |psi2>
        # Rxnabla   = Ra x <psi1| nabla |psi2>
        # Rxnabla_x = Ra_y nabla_z - Ra_z nabla_y
        # Rxnabla_y = Ra_z nabla_x - Ra_x nabla_z
        # Rxnabla_z = Ra_x nabla_y - Ra_y nabla_x
        for v in range(3):
            v1 = (v + 1) % 3
            v2 = (v + 2) % 3
            Rxnabla_ii = (Ra_v[v1] * nabla_iiv[:, :, v2]
                          - Ra_v[v2] * nabla_iiv[:, :, v1])
            dM_caii[v][a][:] = Rxnabla_ii + rxnabla_iiv[:, :, v]

    return dM_caii


def calculate_magnetic_moment_matrix(kpt_u, bfs, correction, r_cG, dM_caii, only_pseudo=False):
    """Calculate magnetic moment matrix in LCAO basis.

    Parameters
    ----------
    kpt_u
        K-points
    bfs
        Basis functions object
    correction
        Correction object
    r_cG
        Grid point coordinates
    dM_caii
        Atomic PAW corrections for magnetic moment
    only_pseudo
        If true, do not add PAW corrections

    Returns
    -------
    Magnetic moment matrix.
    """
    Mstart = correction.Mstart
    Mstop = correction.Mstop
    mynao = Mstop - Mstart
    nao = bfs.Mmax

    assert bfs.Mstart == Mstart
    assert bfs.Mstop == Mstop

    M_cmM = np.zeros((3, mynao, nao), dtype=complex)
    rnabla_cmM = np.empty((3, mynao, nao), dtype=complex)

    for v in range(3):
        v1 = (v + 1) % 3
        v2 = (v + 2) % 3
        rnabla_cmM[:] = 0.0
        bfs.calculate_potential_matrix_derivative(r_cG[v], rnabla_cmM, 0)
        M_cmM[v1] += rnabla_cmM[v2]
        M_cmM[v2] -= rnabla_cmM[v1]

    if not only_pseudo:
        for kpt in kpt_u:
            assert kpt.k == 0
            for c in range(3):
                correction.calculate(kpt.q, dM_caii[c], M_cmM[c],
                                     Mstart, Mstop)

    # The matrix should be real
    assert np.max(np.absolute(M_cmM.imag)) == 0.0
    M_cmM = M_cmM.real.copy()
    return -0.5 * M_cmM


def calculate_magnetic_moment_in_lcao(ksl, rho_mm, M_cmm):
    """Calculate magnetic moment in LCAO.

    Parameters
    ----------
    ksl
        Kohn-Sham Layouts object
    rho_mm
        Density matrix in LCAO basis
    M_cmm
        Magnetic moment matrix in LCAO basis

    Returns
    -------
    Magnetic moment vector.
    """
    assert M_cmm.dtype == float
    mm_c = np.sum(rho_mm.imag * M_cmm, axis=(1, 2))
    if ksl.using_blacs:
        ksl.mmdescriptor.blacsgrid.comm.sum(mm_c)
    return mm_c


class MagneticMomentWriter(TDDFTObserver):
    """Observer for writing time-dependent magnetic moment data.

    The data is written in atomic units.

    The observer attaches to the TDDFT calculator during creation.

    Parameters
    ----------
    paw
        TDDFT calculator
    filename
        File for writing magnetic moment data
    origin
        Origin of the coordinate system used in calculation.
        Possible values:
        ``'COM'``: center of mass (default)
        ``'COC'``: center of cell
        ``'zero'``: (0, 0, 0)
    origin_shift
        Vector in Å shifting the origin from the position defined
        by :attr:`origin`
    calculate_on_grid
        Parameter for testing.
        In LCAO mode, if true, calculation is performed on real-space grid.
        In fd mode, calculation is always performed on real-space grid
        and this parameter is neglected.
    only_pseudo
        Parameter for testing.
        If true, PAW corrections are neglected.
    interval
        Update interval. Value of 1 corresponds to evaluating and
        writing data after every propagation step.
    """
    version = 4

    def __init__(self, paw, filename: str, *,
                 origin: str = None,
                 origin_shift: ArrayLike = None,
                 calculate_on_grid: bool = False,
                 only_pseudo: bool = False,
                 interval: int = 1):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
        if paw.niter == 0:
            if origin is None:
                self.origin = 'COM'
            else:
                self.origin = origin
            self.origin_shift = origin_shift

            # Initialize
            if self.master:
                self.fd = open(filename, 'w')
        else:
            # Read and continue
            self._read_header(filename)
            if self.master:
                self.fd = open(filename, 'a')

            if origin is not None and origin != self.origin:
                raise ValueError('origin changed in restart')
            if origin_shift is not None:
                if self.origin_shift is None \
                   or not np.allclose(origin_shift, self.origin_shift):
                    raise ValueError('origin_shift changed in restart')

        mode = paw.wfs.mode
        assert mode in ['fd', 'lcao'], 'unknown mode: {}'.format(mode)
        if mode == 'fd':
            self.calculate_on_grid = True
        else:
            self.calculate_on_grid = calculate_on_grid

        gd = paw.wfs.gd
        self.timer = paw.timer

        if self.origin == 'COM':
            origin_v = paw.atoms.get_center_of_mass() / Bohr
        elif self.origin == 'COC':
            origin_v = 0.5 * gd.cell_cv.sum(0)
        elif self.origin == 'zero':
            origin_v = np.zeros(3, dtype=float)
        else:
            raise ValueError('unknown origin')
        if self.origin_shift is not None:
            origin_v += np.asarray(self.origin_shift, dtype=float) / Bohr

        R_av = paw.atoms.positions / Bohr - origin_v[np.newaxis, :]
        r_cG, _ = coordinates(gd, origin=origin_v)
        self.origin_v = origin_v

        dM_caii = get_magnetic_moment_atomic_corrections(
                    R_av, paw.setups, paw.hamiltonian.dH_asp.partition)

        if self.calculate_on_grid:
            self.only_pseudo = only_pseudo
            self.r_cG = r_cG
            self.dM_caii = dM_caii

            grad_v = []
            for v in range(3):
                grad_v.append(Gradient(gd, v, dtype=complex, n=2))
            self.grad_v = grad_v
        else:
            M_cmM = calculate_magnetic_moment_matrix(
                             paw.wfs.kpt_u,
                             paw.wfs.basis_functions,
                             paw.wfs.atomic_correction,
                             r_cG, dM_caii,
                             only_pseudo=only_pseudo)

            self.dmat = DensityMatrix(paw)  # XXX
            ksl = paw.wfs.ksl
            if ksl.using_blacs:
                self.M_cmm = ksl.distribute_overlap_matrix(M_cmM)
            else:
                gd.comm.sum(M_cmM)
                self.M_cmm = M_cmM

    def _write(self, line):
        if self.master:
            self.fd.write(line)
            self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = '# %s[version=%s]' % (self.__class__.__name__, self.version)
        line += '('
        args = []
        if self.origin is not None:
            args.append('origin=%s' % repr(self.origin))
        if self.origin_shift is not None:
            args.append('origin_shift=[%.6f, %.6f, %.6f]'
                        % tuple(self.origin_shift))
        line += ', '.join(args)
        line += ')\n'
        line += ('# origin_v = [%.6f, %.6f, %.6f]\n'
                 % tuple(self.origin_v * Bohr))
        line += ('# %15s %22s %22s %22s\n'
                 % ('time', 'cmx', 'cmy', 'cmz'))
        self._write(line)

    def _read_header(self, filename):
        with open(filename, 'r') as f:
            line = f.readline()
        regexp = r"^(?P<name>\w+)\[version=(?P<version>\d+)\]\((?P<args>.*)\)$"
        m = re.match(regexp, line[2:])
        assert m is not None, 'Unknown fileformat'
        assert m.group('name') == self.__class__.__name__
        assert int(m.group('version')) == self.version

        args = m.group('args')
        self.origin = None
        self.origin_shift = None

        m = re.search(r"origin='(\w+)'", args)
        if m is not None:
            self.origin = m.group(1)
        m = re.search(r"origin_shift=\["
                      r"([-+0-9\.]+), "
                      r"([-+0-9\.]+), "
                      r"([-+0-9\.]+)\]",
                      args)
        if m is not None:
            self.origin_shift = [float(m.group(v + 1)) for v in range(3)]

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def _calculate_mm(self, paw):
        if self.calculate_on_grid:
            mm_c = calculate_magnetic_moment_on_grid(
                        paw.wfs, self.grad_v, self.r_cG,
                        self.dM_caii, self.timer,
                        only_pseudo=self.only_pseudo)
        else:
            u = 0
            rho_mm = self.dmat.get_density_matrix((paw.niter, paw.action))[u]
            mm_c = calculate_magnetic_moment_in_lcao(
                        paw.wfs.ksl, rho_mm, self.M_cmm)
        assert mm_c.shape == (3,)
        assert mm_c.dtype == float
        return mm_c

    def _write_mm(self, paw):
        time = paw.time
        mm = self._calculate_mm(paw)
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
