import numpy as np


def select_kpts(kpts, kd):
    """Function to process input parameters that take a list of k-points given
    in different format and returns a list of indices of the corresponding
    k-points in the IBZ."""

    if kpts is None:
        # Do all k-points in the IBZ:
        return np.arange(kd.nibzkpts)

    if np.asarray(kpts).ndim == 1:
        return kpts

    # Find k-points:
    bzk_Kc = kd.bzk_kc
    indices = []
    for k_c in kpts:
        d_Kc = bzk_Kc - k_c
        d_Kc -= d_Kc.round()
        K = abs(d_Kc).sum(1).argmin()
        if not np.allclose(d_Kc[K], 0):
            raise ValueError('Could not find k-point: {k_c}'
                             .format(k_c=k_c))
        k = kd.bz2ibz_k[K]
        indices.append(k)
    return indices
