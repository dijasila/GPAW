from __future__ import annotations
import numpy as np
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.matrix import Matrix, create_distribution
from gpaw.core.plane_waves import PlaneWaveAtomCenteredFunctions, PlaneWaves
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.new.calculation import DFTState


def pw_matrix(pw: PlaneWaves,
              pt_aiG: PlaneWaveAtomCenteredFunctions,
              dH_aii: AtomArrays,
              dS_aii: AtomArrays,
              vt_R: UniformGridFunctions,
              comm) -> tuple[Matrix, Matrix]:
    """

    :::

                 _ _     _ _
            /  -iG.r ~  iG.r _
      O   = | e      O e    dr
       GG'  /

    :::

      ~   ^   ~ _    _ _     --- ~a _ _a    a  ~  _  _a
      H = T + v(r) δ(r-r') + <   p (r-R ) ΔH   p (r'-R )
                             ---  i         ij  j
                             aij

    :::

      ~     _ _     --- ~a _ _a    a  ~  _  _a
      S = δ(r-r') + <   p (r-R ) ΔS   p (r'-R )
                    ---  i         ij  j
                    aij
    """
    assert pw.dtype == complex
    npw = pw.shape[0]
    dist = create_distribution(npw, npw, comm, -1, 1)
    H_GG = dist.matrix()
    S_GG = dist.matrix()
    G1, G2 = dist.my_row_range()

    x_G = pw.zeros()
    x_R = vt_R.desc.new(dtype=complex).zeros()
    N = x_R.data.size
    dv = pw.dv / N

    for G in range(G1, G2):
        x_G.data[G] = 1.0
        x_G.ifft(out=x_R)
        x_R *= vt_R
        x_R.fft(out=x_G)
        H_GG.data[G - G1] = dv * x_G.data
        x_G.data[G] = 0.0

    H_GG.add_to_diagonal(dv * pw.ekin_G)

    S_GG.data[:] = 0.0
    S_GG.add_to_diagonal(dv)

    f_GI = pt_aiG._lfc.expand()
    print(f_GI.shape)
    nI = f_GI.shape[1]
    dH_II = np.zeros((nI, nI))
    dS_II = np.zeros((nI, nI))
    I1 = 0
    for a, dH_ii in dH_aii.items():
        dS_ii = dS_aii[a]
        I2 = I1 + len(dS_ii)
        dH_II[I1:I2, I1:I2] = dH_ii / N**2
        dS_II[I1:I2, I1:I2] = dS_ii / N**2
        I1 = I2

    H_GG.data += np.dot(f_GI[G1:G2].conj(), np.dot(dH_II, f_GI.T))
    S_GG.data += np.dot(f_GI[G1:G2].conj(), np.dot(dS_II, f_GI.T))

    return H_GG, S_GG


def diagonalize_full_hamiltonian(self, ham, atoms, log,
                                 nbands=None, ecut=None, scalapack=None,
                                 expert=False):

    if self.dtype != complex:
        raise ValueError(
            'Please use mode=PW(..., force_complex_dtype=True)')

    if self.gd.comm.size > 1:
        raise ValueError(
            "Please use parallel={'domain': 1}")

    S = self.bd.comm.size

    if nbands is None and ecut is None:
        nbands = pd.ngmin // S * S
    elif nbands is None:
        ecut /= Ha
        vol = abs(np.linalg.det(self.gd.cell_cv))
        nbands = int(vol * ecut**1.5 * 2**0.5 / 3 / pi**2)

    if nbands % S != 0:
        nbands += S - nbands % S

    assert nbands <= pd.ngmin

    if expert:
        iu = nbands
    else:
        iu = None

    self.bd = bd = BandDescriptor(nbands, self.bd.comm)
    self.occupations.bd = bd

    log('Diagonalizing full Hamiltonian ({} lowest bands)'.format(nbands))
    log('Matrix size (min, max): {}, {}'.format(pd.ngmin,
                                                pd.ngmax))
    mem = 3 * pd.ngmax**2 * 16 / S / 1024**2
    log('Approximate memory used per core to store H_GG, S_GG: {:.3f} MB'
        .format(mem))
    log('Notice: Up to twice the amount of memory might be allocated\n'
        'during diagonalization algorithm.')
    log('The least memory is required when the parallelization is purely\n'
        'over states (bands) and not k-points, set '
        "GPAW(..., parallel={'kpt': 1}, ...).")

    if S > 1:
        if isinstance(scalapack, (list, tuple)):
            nprow, npcol, b = scalapack
            assert nprow * npcol == S, (nprow, npcol, S)
        else:
            nprow = int(round(S**0.5))
            while S % nprow != 0:
                nprow -= 1
            npcol = S // nprow
            b = 64
        log('ScaLapack grid: {}x{},'.format(nprow, npcol),
            'block-size:', b)
        bg = BlacsGrid(bd.comm, S, 1)
        bg2 = BlacsGrid(bd.comm, nprow, npcol)
        scalapack = True
    else:
        scalapack = False

    self.set_positions(atoms.get_scaled_positions())
    self.kpt_u[0].projections = None
    self.allocate_arrays_for_projections(self.pt.my_atom_indices)

    myslice = bd.get_slice()

    pb = ProgressBar(log.fd)
    nkpt = len(self.kpt_u)

    for u, kpt in enumerate(self.kpt_u):
        pb.update(u / nkpt)
        npw = len(pd.Q_qG[kpt.q])
        if scalapack:
            mynpw = -(-npw // S)
            md = BlacsDescriptor(bg, npw, npw, mynpw, npw)
            md2 = BlacsDescriptor(bg2, npw, npw, b, b)
        else:
            md = md2 = MatrixDescriptor(npw, npw)

        with self.timer('Build H and S'):
            H_GG, S_GG = self.hs(ham, kpt.q, kpt.s, md)

        if scalapack:
            r = Redistributor(bd.comm, md, md2)
            H_GG = r.redistribute(H_GG)
            S_GG = r.redistribute(S_GG)

        psit_nG = md2.empty(dtype=complex)
        eps_n = np.empty(npw)

        with self.timer('Diagonalize'):
            if not scalapack:
                md2.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n,
                                           iu=iu)
            else:
                md2.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n)
        del H_GG, S_GG

        kpt.eps_n = eps_n[myslice].copy()

        if scalapack:
            md3 = BlacsDescriptor(bg, npw, npw, bd.maxmynbands, npw)
            r = Redistributor(bd.comm, md2, md3)
            psit_nG = r.redistribute(psit_nG)

        kpt.psit = PlaneWaveExpansionWaveFunctions(
            self.bd.nbands, pd, self.dtype,
            psit_nG[:bd.mynbands].copy(),
            kpt=kpt.q, dist=(self.bd.comm, self.bd.comm.size),
            spin=kpt.s, collinear=self.collinear)
        del psit_nG

        with self.timer('Projections'):
            self.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        kpt.f_n = None

    pb.finish()

    self.calculate_occupation_numbers()

    return nbands


def diagonalize(state: DFTState,
                nbands: int) -> DFTState:
    vt_sR = state.potential.vt_sR
    dH_asii = state.potential.dH_asii
    dS_aii = [delta_iiL[:, :, 0]
              for delta_iiL in state.density.delta_aiiL]
    for wfs in state.ibzwfs:
        H_GG, S_GG = pw_matrix(wfs.psit_nX.desc,
                               wfs.pt_aiX,
                               dH_asii[:, wfs.spin],
                               dS_aii,
                               vt_sR[wfs.spin],
                               wfs.psit_nX.comm)
        C_nG = H_GG.eigh(S_GG)
    return state
