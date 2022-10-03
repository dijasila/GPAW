"""

Numerical backend for GW calculations. In preparations to create numerical
GPU kernels, all numerical methods are to be processed through GWBackend.,

"""

class GWBackend:
    def __init__(self):
        pass

    def spectral_update(self, m_w, p1_m, p2_m, n_mG, A_wGG):
        """
            m_w: Spesification that from which pair density each frequency
            point will end to (exclusive). Essentially this is a packed row
            format representation of the sparse interpolation matrix.

            p1_m: Interpolation coefficient for the pair density outer product to frequency point w.
            p2_m: Interpolation coefficient for the pair density outer product to frequency point w+1.

            n_mG: Pair densities

        """
        raise NotImplementedError


def GWBackendCPUReferenceImplementation:
    def __init__(self):
        GWBackend.__init__(self)

    def spectral_update(self, m_w, p1_m, p2_m, n_mG, A_wGG):
        nw = len(A_wGG)
        mstart = 0
        for w, mend in enumerate(m_w):
            nslice_mG = n_mG[mstart:mend]
            A_wGG[w] += (p1_m * n_mG.T.conj()) @ n_mG
            if w+1 <= nw:
                A_wGG[w+1] += (p2_m * n_mG.T.conj()) @ n_mG
            mstart = mend

    def inplace_hilbert_transform(self, H_ww, A_wGG):
        nw = len(A_wGG)
        A_wGG[:] = (H_ww @ A_wGG.reshape((nw, -1))).reshape(*A_wGG.shape)

    def pair_density_generation(self, psit_iG, psit_aG, eps_i, eps_a, P_ani, Ghole_G, Gel_G, Gpair_G):
        # ifft(psit_iG) * ifft

