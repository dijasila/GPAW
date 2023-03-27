from ase.io.ulm import ulmopen


def test_fe_fixture(gpw_files):
    """Make sure there is a gap between the converged bands
    and the next band.
    """
    with ulmopen(gpw_files['fe_pw']) as reader:
        eig_skn = reader.wave_functions.eigenvalues
        nconv = reader.parameters.convergence['bands']
    gap_sk = eig_skn[:, :, nconv] - eig_skn[:, :, nconv - 1]
    assert gap_sk.min() > 0.02
