from collections import namedtuple


KPoint = namedtuple(
    'KPoint',
    ['psit',   # plane-wave expansion of wfs
     'proj',   # projections
     'f_n',    # occupations numbers between 0 and 1
     'k_c',    # k-vector in units of reciprocal cell
     'weight'  # weight of k-point
     ])

RSKPoint = namedtuple(
    'RealSpaceKPoint',
    ['u_nR',  # wfs on a real-space grid
     'proj',  # same as above
     'f_n',   # ...
     'k_c',
     'weight',
     # 'index'  # IBZ k-point index
     ])


def to_real_space(psit, na=0, nb=None):
    pd = psit.pd
    comm = pd.comm
    S = comm.size
    q = psit.kpt
    nbands = len(psit.array)
    nb = nb or nbands
    u_nR = pd.gd.empty(nbands, pd.dtype, global_array=True)
    for n1 in range(0, nbands, S):
        n2 = min(n1 + S, nbands)
        u_G = pd.alltoall1(psit.array[n1:n2], q)
        if u_G is not None:
            n = n1 + comm.rank
            u_nR[n] = pd.ifft(u_G, local=True, safe=False, q=q)
        for n in range(n1, n2):
            comm.broadcast(u_nR[n], n - n1)

    return u_nR[na:nb]


def get_kpts(wfs, spin, nocc=-1):
    assert wfs.world.size == wfs.gd.comm.size
    kd = wfs.kd
    K = kd.nibzkpts
    k1 = spin * K
    k2 = k1 + K
    kpts = []
    for kpt in wfs.mykpts[k1:k2]:
        psit = kpt.psit
        proj = get_projections(wfs, spin, k, nocc)
        if nocc != -1:
            psit = psit.view(0, nocc)
            proj = proj.view(0, nocc)
        kpt = KPoint(psit,
                     proj,
                     kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                     kd.ibzk_kc[kpt.k],
                     kd.weight_k[kpt.k])
        kpts.append(kpt)
    return kpts


def get_projections
kpt.projections