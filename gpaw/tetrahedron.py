import numpy as np
from scipy.spatial import Delaunay


def a1(e1, e2, e3, e4):
    x = 1.0 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    return -e1**3 * x


def a2(e1, e2, e3, e4):
    x = 1.0 / ((e3 - e1) * (e4 - e1))
    return x * ((e2 - e1)**2
                - 3 * (e2 - e1) * e2
                + 3 * e2**2
                + (e3 - e1 + e4 - e2) / ((e3 - e2) * (e4 - e2)) * e2**3)


def a3(e1, e2, e3, e4):
    x = 1.0 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    return 1 - x * e4**3


def f(size_c, i_k, rcell_cv, eig_in, ef):
    nbzk, = i_k.shape
    nibzk, nbands = eig_in.shape

    assert size_c.shape == (3,)
    assert rcell_cv.shape == (3, 3)
    assert i_k.max() == nibzk - 1
    assert nbzk == size_c.prod()

    eig_in = eig_in - ef
    nocc_i = (eig_in < 0.0).sum(axis=1)
    n1 = nocc_i.min()
    n2 = nocc_i.max()

    ne = n1
    dnedef = 0.0

    if n1 == n2:
        return ne, dnedef

    # Find the 6 tetrahedra:
    k012_sc = np.array([[i, j, k]
                        for i in [0, 1] for j in [0, 1] for k in [0, 1]])
    dt = Delaunay(k012_sc.dot(rcell_cv / size_c[:, np.newaxis]))
    s_tq = dt.simplices
    assert s_tq.shape == (6, 4)
    k012_tqc = k012_sc[s_tq]

    k012_ck = np.unravel_index(np.arange(nbzk), size_c)
    k012_tqck = k012_tqc[..., np.newaxis] + k012_ck
    k012_cktq = np.transpose(k012_tqck, (2, 3, 0, 1))
    k_ktq = np.ravel_multi_index(k012_cktq.reshape((3, nbzk * 6 * 4)),
                                 size_c,
                                 mode='wrap').reshape((nbzk, 6, 4))
    i_ktq = i_k[k_ktq]

    eig_Tq = eig_in[i_ktq, n1:n2].transpose((0, 1, 3, 2)).reshape(
        (nbzk * 6 * (n2 - n1), 4))
    eig_Tq.sort(axis=1)

    eig_Tq = eig_Tq[eig_Tq[:, 0] < 0.0]

    mask1_T = eig_Tq[:, 1] > 0.0
    ne += a1(*eig_Tq[mask1_T].T).sum() / nbzk / 6

    mask2_T = ~mask1_T & (eig_Tq[:, 2] > 0.0)
    ne += a2(*eig_Tq[mask2_T].T).sum() / nbzk / 6

    mask3_T = ~mask1_T & ~mask2_T & (eig_Tq[:, 3] > 0.0)
    ne += a3(*eig_Tq[mask3_T].T).sum() / nbzk / 6

    mask4_T = ~mask1_T & ~mask2_T & ~mask3_T
    ne += mask4_T.sum() / nbzk / 6

    return ne, 0.0


size_c = np.array([2, 2, 1])
i_k = np.array([0, 1, 2, 1])
rcell_cv = np.diag([1.0, 1.0, 0.1])
eig_in = np.array([[1.0], [0.0], [0.0]])

X = np.linspace(-0.5, 1.5, 250)
import matplotlib.pyplot as plt
plt.plot(X, [f(size_c, i_k, rcell_cv, eig_in, ef)[0] for ef in X])
e = np.array([0, 0.33, 0.66, 1.0])
#plt.plot(X, [a1(*(e - x)) for x in X], label='1')
#plt.plot(X, [a2(*(e - x)) for x in X], label='2')
#plt.plot(X, [a3(*(e - x)) for x in X], label='3')
#plt.legend()
plt.show()
