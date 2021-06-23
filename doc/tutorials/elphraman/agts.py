from myqueue.workflow import run


def workflow():
    ph = run(script='phonon.py')
    el = run(script='elph.py')
    mo = run(script='momentum_matrix.py')
    with ph, el:
        with run(script='supercell_matrix.py'):
            with run(script='elph_matrix.py', cores=1), mo:
                run(script='raman_intensities.py', cores=1)
                run(function=check)


def check():
    """Read result and make sure it's OK."""
    import numpy as np
    ri = np.load("RI_xz.npy")
    assert ri[1].argmax == 1304
    assert np.isclose(max(ri[1]), 0.04889, atol=1e-3)
