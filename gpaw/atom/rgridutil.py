import numpy as np
from math import pi
from gpaw.atom.aeatom import Channel


def create_basis_function(l, n, tailnorm, scale, rgd, waves, vtr_g):
    # Find cutoff radii:
    n_g = np.add.accumulate(waves.phit_ng[n]**2 * rgd.r_g**2 * rgd.dr_g)
    norm = n_g[-1]
    g2 = (norm - n_g > tailnorm * norm).sum()
    r2 = rgd.r_g[g2]
    r1 = max(0.6 * r2, waves.rcut)
    g1 = rgd.ceil(r1)
    # Set up confining potential:
    r = rgd.r_g[g1:g2]
    vtr_g = vtr_g.copy()
    vtr_g[g1:g2] += scale * np.exp((r2 - r1) / (r1 - r)) / (r - r2)**2
    vtr_g[g2:] = np.inf

    # Nonlocal PAW stuff:
    pt_ng = waves.pt_ng
    dH_nn = waves.dH_nn
    dS_nn = waves.dS_nn
    N = len(pt_ng)

    u_g = rgd.zeros()
    u_ng = rgd.zeros(N)
    duodr_n = np.empty(N)
    a_n = np.empty(N)

    e = waves.e_n[n]
    e0 = e
    ch = Channel(l)
    while True:
        duodr, a = ch.integrate_outwards(u_g, rgd, vtr_g, g1, e)

        for n in range(N):
            duodr_n[n], a_n[n] = ch.integrate_outwards(u_ng[n], rgd,
                                                       vtr_g, g1, e,
                                                       pt_g=pt_ng[n])

        A_nn = (dH_nn - e * dS_nn) / (4 * pi)
        B_nn = rgd.integrate(pt_ng[:, None] * u_ng, -1)
        c_n = rgd.integrate(pt_ng * u_g, -1)
        d_n = np.linalg.solve(np.dot(A_nn, B_nn) + np.eye(N),
                              np.dot(A_nn, c_n))
        u_g[:g1 + 1] -= np.dot(d_n, u_ng[:, :g1 + 1])
        a -= np.dot(d_n, a_n)
        duodr -= np.dot(duodr_n, d_n)
        uo = u_g[g1]

        duidr = ch.integrate_inwards(u_g, rgd, vtr_g, g1, e, gmax=g2)
        ui = u_g[g1]
        A = duodr / uo - duidr / ui
        u_g[g1:] *= uo / ui
        x = (norm / rgd.integrate(u_g**2, -2) * (4 * pi))**0.5
        u_g *= x
        a *= x

        if abs(A) < 1e-5:
            break

        e += 0.5 * A * u_g[g1]**2

    u_g[1:] /= rgd.r_g[1:]
    u_g[0] = a * 0.0**l
    return u_g, r1, r2, e - e0
