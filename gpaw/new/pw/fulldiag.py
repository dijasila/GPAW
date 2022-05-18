import numpy as np


def pw_matrix(pw, gcomm, dH_aii, vt_R=None):
    pd = None

    npw = len(pd.Q_qG[q])
    N = pd.tmp_R.size

    if md is None:
        H_GG = np.zeros((npw, npw), complex)
        S_GG = np.zeros((npw, npw), complex)
        G1 = 0
        G2 = npw
    else:
        H_GG = md.zeros(dtype=complex)
        S_GG = md.zeros(dtype=complex)
        if S_GG.size == 0:
            return H_GG, S_GG
        G1, G2 = next(md.my_blocks(S_GG))[:2]

    H_GG.ravel()[G1::npw + 1] = (0.5 * pd.gd.dv / N *
                                 pd.G2_qG[q][G1:G2])
    for G in range(G1, G2):
        x_G = pd.zeros(q=q)
        x_G[G] = 1.0
        H_GG[G - G1] += (pd.gd.dv / N *
                         pd.fft(ham.vt_sG[s] *
                                pd.ifft(x_G, q), q))

    S_GG.ravel()[G1::npw + 1] = pd.gd.dv / N

    f_GI = pt.expand(q)
    nI = f_GI.shape[1]
    dH_II = np.zeros((nI, nI))
    dS_II = np.zeros((nI, nI))
    I1 = 0
    for a in self.pt.my_atom_indices:
        dH_ii = unpack(ham.dH_asp[a][s])
        dS_ii = setups[a].dO_ii
        I2 = I1 + len(dS_ii)
        dH_II[I1:I2, I1:I2] = dH_ii / N**2
        dS_II[I1:I2, I1:I2] = dS_ii / N**2
        I1 = I2

    H_GG += np.dot(f_GI[G1:G2].conj(), np.dot(dH_II, f_GI.T))
    S_GG += np.dot(f_GI[G1:G2].conj(), np.dot(dS_II, f_GI.T))

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
