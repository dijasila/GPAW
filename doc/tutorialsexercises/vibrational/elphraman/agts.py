from myqueue.workflow import run


def workflow():
    # Step 1: Finite displacement run in supercell
    disp = run(script='displacement.py', tmax='10h', cores=48)
    # Step 1a: Obtain unit cell wavefunctions (LCAO)
    scf = run(script='scf.py', cores=24, tmax='15m')
    with disp:
        # Step 2: Project derivative onto LCAO orbitals
        sc = run(script='supercell.py', tmax='1h', cores=4)
    with disp:
        # Step 3a: Extract phonon modes
        ph = run(script='phonons.py', cores=1)
    with scf:
        # Step 3b: Extract dipole moments
        dip = run(script='dipolemoment.py', cores=1)
    with disp, sc, scf, ph:
        run(script='gmatrix.py', cores=1)
    with disp, sc, scf, dip, ph:
        ra = run(script='raman.py', cores=1)
        with ra:
            run(script='plot_spectrum.py', cores=1)
            # run(function=check)

# def check():
    # """Read result and make sure it's OK."""
    # import numpy as np
    # from gpaw.test import findpeak
    # ri = np.load('RI_xz_632nm.npy')
    # x0, y0 = findpeak(ri[0], ri[1])
    # print(x0, y0)
    # assert np.isclose(x0, 1304.497, atol=0.2)
    # assert np.isclose(y0, 0.044, atol=0.01)
