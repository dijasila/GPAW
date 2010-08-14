import numpy as np
from gpaw.xc.libxc import LibXC, short_names
from gpaw.xc.kernel import XCKernel, codes
from gpaw.xc.bee import BEE1

functionals = [LibXC(name) for name in short_names]
functionals += [XCKernel(name) for name in codes]
functionals += [BEE1()]

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
    [[0.1, 0.2, 0.1, -0.08, 0.10, 0.01, 0.05],
     [0.1, 0.1, 0.1,  0.01, 0.01, 0.01, 0.01],
     [0.1, 0.1, 0.1,  0.15, 0.20, 0.01, 0.05]]).T.copy()

eps = 1.0e-5

for xc in functionals:
    e0_g, d0_xg = f2(n_xg, xc)
    d_xg = np.empty_like(d0_xg)
    for x, n_g in enumerate(n_xg):
        m_xg = n_xg.copy()
        m_xg[x] += eps
        d_xg[x] = 0.5 * f2(m_xg, xc)[0] / eps
        m_xg[x] -= 2 * eps
        d_xg[x] -= 0.5 * f2(m_xg, xc)[0] / eps
    print xc.name, abs(d0_xg-d_xg).max()
    print d0_xg-d_xg
    
n_xg = np.array(
    [[0.2, 0.1, 0.5],
     [0.01, 0.01, 0.1 ],
     [0.1, 0.3, 0.5]]).T.copy()

for xc in functionals:
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
    es_g, ds_xg = f2(ns_xg, xc)
    print xc.name, abs(d0_xg-d_xg).max(), abs(es_g - e0_g).max()
    print abs(ds_xg[:2] - d0_xg[0]).max()
    print abs(ds_xg[2:5].sum(0) / 4 - d0_xg[1]).max()
    print abs(ds_xg[5:] - d0_xg[2]).max()
