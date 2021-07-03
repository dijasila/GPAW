from myqueue.workflow import run


def workflow():
    ph = run(script='phonon.py')
    el = run(script='elph.py')
    mo = run(script='momentum_matrix.py')
    with ph, el:
        with run(script='supercell_matrix.py'):
            with run(script='elph_matrix.py'), mo:
                run(script='raman_intensities.py')
                run(function=check)


def check():
    """Read result and make sure it's OK."""
    import numpy as np
    ri = np.load("RI_xz_632nm.npy")
    assert ri[1].argmax() == 1304
    assert np.isclose(max(ri[1]), 4.8486e-5, rtol=1e-3)
