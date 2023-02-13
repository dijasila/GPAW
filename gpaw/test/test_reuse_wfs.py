import pytest
from ase.build import bulk

from gpaw import GPAW, PW, Mixer
from gpaw.convergence_criteria import Eigenstates


class MyConvergenceCriterion(Eigenstates):
    def __init__(self, tol):
        super().__init__(tol)
        self.history = []

    def get_error(self, context):
        value = super().get_error(context)
        self.history.append(value)
        return value


def run(mode, method, setups):
    atoms = bulk('Si')
    atoms.rattle(stdev=0.01, seed=17)  # Break symmetry

    kwargs = {}
    if mode == 'pw':
        kwargs.update(mode=PW(200.0))

    conv_tol = 1e-9
    conv = MyConvergenceCriterion(conv_tol)

    calc = GPAW(mixer=Mixer(0.4, 5, 20.0),
                txt=None,
                setups=setups,
                convergence={'custom': [conv]},
                experimental={'reuse_wfs_method': method},
                xc='PBE',
                kpts=[2, 2, 2],
                **kwargs)

    atoms.calc = calc
    E1 = atoms.get_potential_energy()
    assert conv.history[-1] < conv_tol
    niter1 = len(conv.history)
    del conv.history[:]

    atoms.rattle(stdev=0.001)

    E2 = atoms.get_potential_energy()
    niter2 = len(conv.history)
    reuse_error = conv.history[0]

    # If the energy is exactly or suspiciously close to zero, it's because
    # nothing was done at all (something was cached but shouldn't have been)
    delta_e = abs(E2 - E1)
    assert delta_e > 1e-6, delta_e
    return niter1, niter2, reuse_error


@pytest.mark.later
@pytest.mark.parametrize('mode, reuse_type, setups, max_reuse_error', [
    ('pw', 'paw', 'paw', 1e-5),
    ('pw', None, 'paw', 1e-4),
    ('fd', 'paw', 'paw', 1e-4),
    ('fd', None, 'paw', 1e-3),
    # ('pw', 'paw', 'sg15', 0)
])
def test_reuse_wfs(mode, reuse_type, setups, max_reuse_error):
    """Check that wavefunctions are meaningfully reused.

    For a different modes and parameters, this test asserts that the
    initial wavefunction error in the second scf step is below a
    certain threshold, indicating that we are doing better than if
    we started from scratch."""

    niter_first, niter_second, reuse_error = run(mode, reuse_type, setups)

    # It should at the very least be faster to do the second step:
    assert niter_second < niter_first
    assert reuse_error < max_reuse_error
