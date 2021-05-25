import numpy as np
from math import pi
from ase.units import Hartree
from gpaw.atom.aeatom import Channel
from gpaw.basis_data import Basis, BasisFunction


"""
TODO: Make this function work:

def create_upf_basis(setup):
    variables = ...
    return create_basis_function(...)


From the UPF setup we have (easily) l, n, and any grid (rgd) we want.
 * vtr_g is probably setup.vbar_g or vbar_g * r or something like that.
 * tailnorm: We want to replace that with a confinement energy.  Later.
 * waves.phit_ng: Not really needed
 * n_g: Instead of using n_g, use another way to confine the states.
 * waves.rcut: the cutoff
 * waves.pt_ng: the projectors, we have those on the setup
 * waves.dH_nn: exist on the setup.
   It's setup.K_p aka setupdata.expand_hamiltonian_matrix().
 * waves.dS_nn: probably zero
 * waves.e_n: When we switch to using confinement energy instead of tailnorm,
   we no longer need to depend on e_n.  So we should do that almost now.
"""


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


def create_basis_set(*, tailnorm=0.0005, scale=200.0, splitnorm=0.16,
                     rgd, symbol, waves_l, vtr_g, log, nvalence):
    basis = Basis(symbol, 'dzp', readxml=False, rgd=rgd)

    # We print text to sdtout and put it in the basis-set file
    txt = 'Basis functions:\n'

    # Bound states:
    for l, waves in enumerate(waves_l):
        for i, n in enumerate(waves.n_n):
            if n > 0:
                tn = tailnorm
                if waves.f_n[i] == 0:
                    tn = min(0.05, tn * 20)  # no need for long tail
                phit_g, ronset, rc, de = create_basis_function(
                    l, i, tn, scale, rgd=rgd, waves=waves_l[l],
                    vtr_g=vtr_g)
                bf = BasisFunction(n, l, rc, phit_g, 'bound state')
                basis.append(bf)

                txt += '%d%s bound state:\n' % (n, 'spdf'[l])
                txt += ('  cutoff: %.3f to %.3f Bohr (tail-norm=%f)\n' %
                        (ronset, rc, tn))
                txt += '  eigenvalue shift: %.3f eV\n' % (de * Hartree)

    # Split valence:
    for l, waves in enumerate(waves_l):
        # Find the largest n that is occupied:
        n0 = None
        for f, n in zip(waves.f_n, waves.n_n):
            if n > 0 and f > 0:
                n0 = n
        if n0 is None:
            continue

        for bf in basis.bf_j:
            if bf.l == l and bf.n == n0:
                break

        # Radius and l-value used for polarization function below:
        rcpol = bf.rc
        lpol = l + 1

        phit_g = bf.phit_g

        # Find cutoff radius:
        n_g = np.add.accumulate(phit_g**2 * rgd.r_g**2 * rgd.dr_g)
        norm = n_g[-1]
        gc = (norm - n_g > splitnorm * norm).sum()
        rc = rgd.r_g[gc]

        phit2_g = rgd.pseudize(phit_g, gc, l, 2)[0]  # "split valence"
        bf = BasisFunction(n, l, rc, phit_g - phit2_g, 'split valence')
        basis.append(bf)

        txt += '%d%s split valence:\n' % (n0, 'spdf'[l])
        txt += '  cutoff: %.3f Bohr (tail-norm=%f)\n' % (rc, splitnorm)

    # Polarization:
    gcpol = rgd.round(rcpol)
    alpha = 1 / (0.25 * rcpol)**2

    # Gaussian that is continuous and has a continuous derivative at rcpol:
    phit_g = np.exp(-alpha * rgd.r_g**2) * rgd.r_g**lpol
    phit_g -= rgd.pseudize(phit_g, gcpol, lpol, 2)[0]
    phit_g[gcpol:] = 0.0

    bf = BasisFunction(None, lpol, rcpol, phit_g, 'polarization')
    basis.append(bf)
    txt += 'l=%d polarization functions:\n' % lpol
    txt += '  cutoff: %.3f Bohr (r^%d exp(-%.3f*r^2))\n' % (rcpol, lpol,
                                                            alpha)

    log(txt)

    # Write basis-set file:
    basis.generatordata = txt
    basis.generatorattrs.update(dict(tailnorm=tailnorm,
                                     scale=scale,
                                     splitnorm=splitnorm))
    basis.name = '%de.dzp' % nvalence
    return basis
