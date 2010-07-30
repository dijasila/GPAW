import numpy as np
from gpaw.xc.libxc import LibXC, short_names

def f1(n_xg, xc):
    e_g = np.empty_like(n_xg[0])
    n_sg = n_xg[:1]
    sigma_xg = n_xg[1:2]
    tau_sg = n_xg[2:]
    dedn_sg = np.zeros_like(n_sg)
    dedsigma_xg = np.zeros_like(sigma_xg)
    dedtau_sg = np.zeros_like(tau_sg)
    xc.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg)
    return e_g, np.concatenate((dedn_sg, dedsigma_xg, dedtau_sg))

def f2(n_xg, xc):
    e_g = np.empty_like(n_xg[0])
    n_sg = n_xg[:2]
    sigma_xg = n_xg[2:5]
    tau_sg = n_xg[5:]
    dedn_sg = np.zeros_like(n_sg)
    dedsigma_xg = np.zeros_like(sigma_xg)
    dedtau_sg = np.zeros_like(tau_sg)
    xc.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg)
    return e_g, np.concatenate((dedn_sg, dedsigma_xg, dedtau_sg))

n_xg = np.array(
    [[0.1, 0.2, 0.1, -0.08, 0.1, 0.01, 0.05],
     [0.1, 0.1, 0.1,  0.0 , 0.0, 0.01, 0.0 ],
     [0.1, 0.1, 0.1,  0.15, 0.2, 0.01, 0.05]]).T.copy()

eps = 1.0e-5

for name in short_names:
    xc = LibXC(name)
    e0_g, d0_xg = f2(n_xg, xc)
    d_xg = np.empty_like(d0_xg)
    for x, n_g in enumerate(n_xg):
        m_xg = n_xg.copy()
        m_xg[x] += eps
        d_xg[x] = 0.5 * f2(m_xg, xc)[0] / eps
        m_xg[x] -= 2 * eps
        d_xg[x] -= 0.5 * f2(m_xg, xc)[0] / eps
    print name, abs(d0_xg-d_xg).max()

n_xg = np.array(
    [[0.2, 0.1, 0.005],
     [0.1, 0.01, 0.00 ],
     [0.1, 0.3, 0.005]]).T.copy()

for name in short_names:
    xc = LibXC(name)
    e0_g, d0_xg = f1(n_xg, xc)
    d_xg = np.empty_like(d0_xg)
    for x, n_g in enumerate(n_xg):
        m_xg = n_xg.copy()
        m_xg[x] += eps
        d_xg[x] = 0.5 * f1(m_xg, xc)[0] / eps
        m_xg[x] -= 2 * eps
        d_xg[x] -= 0.5 * f1(m_xg, xc)[0] / eps
    ns_xg = np.empty((7, len(n_g)))
    ns_xg[:2] = n_xg[0] / 2
    ns_xg[2:5] = n_xg[1] / 4
    ns_xg[5:] = n_xg[2] / 2
    es_g = f2(ns_xg, xc)[0]
    print name, abs(d0_xg-d_xg).max(), abs(es_g - e0_g).max()
    if name == 'TPSS':
        print abs(d0_xg-d_xg);sdfg
