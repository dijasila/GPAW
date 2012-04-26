import numpy as np
import ase.units as units

from gpaw.wavefunctions.pw import ft


def stress(calc):
    wfs = calc.wfs
    dens = calc.density
    ham = calc.hamiltonian
    ghat = dens.ghat

    volume = calc.atoms.get_volume() / units.Bohr**3

    pkin = calc.wfs.get_kinetic_stress().trace() / volume
    pxc = 3 * calc.hamiltonian.stress / volume

    S_q = 0.0
    for a, emiGR_G in enumerate(ghat.emiGR_Ga.T):
        S_q += dens.Q_aL[a][0] * emiGR_G

    spline = wfs.setups[0].ghat_l[0]
    f = ft(spline)

    pd = dens.pd3
    G_q = pd.G2_qG[0]**0.5
    G_q[0] = 1.0

    dfdGoG_q = np.array([f.get_value_and_derivative(G)[1] / G
                         for G in G_q])

    x = (4 * np.pi)**-0.5 *4*np.pi/volume**2* pd.gd.N_c.prod()

    epot = pd.integrate(dens.rhot_q / G_q**2, dens.rhot_q) * 4 * np.pi
    ppot = -pd.integrate(S_q * dfdGoG_q, dens.rhot_q) * x

    print epot, ham.epot, volume, dens.rhot_q[0], pd.G2_qG[0,0],pd.gd.N_c.prod()
    ppot2 = -ham.epot / volume




    S_q = 0.0
    for a, emiGR_G in enumerate(ham.vbar.emiGR_Ga.T):
        S_q += emiGR_G

    spline = wfs.setups[0].vbar
    f = ft(spline)

    pd = dens.pd2
    G_q = pd.G2_qG[0]**0.5

    GdfdG_q = np.array([f.get_value_and_derivative(G)[1] * G
                         for G in G_q])

    x = (4 * np.pi)**-0.5 /volume**2* pd.gd.N_c.prod()

    pbar = -pd.integrate(S_q * GdfdG_q, dens.nt_sQ[0]) * x - 3 * ham.ebar / volume



    print volume,pkin,pxc,ppot,ppot2
    return pkin+pxc+ppot+ppot2+pbar
