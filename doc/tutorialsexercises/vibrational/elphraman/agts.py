from myqueue.workflow import run


def workflow():
    ph = run(script='phonon.py')
    el = run(script='elph.py', tmax='1h')
    mo = run(script='momentum_matrix.py')
    with ph, el:
        with run(script='supercell_matrix.py', tmax='1h'):
            with run(script='elph_matrix.py'), mo:
                with run(script='raman_intensities.py'):
                    run(function=check)


def check():
    """Read result and make sure it's OK."""
    import numpy as np
    from gpaw.test import findpeak
    ri = np.load('RI_xz_632nm.npy')
    x0, y0 = findpeak(ri[0], ri[1])
    assert np.isclose(x0, 1304.497, atol=0.2)
    assert np.isclose(y0, 5e-5, atol=1e-5)
