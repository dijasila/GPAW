import pytest

from gpaw.response.goldstone import find_root


def parabolic_function(lambd):
    """Define a parabolic function f(λ) with a root nearby λ=1.

    Defining,

    f(λ) = aλ² + bλ + c

    we need the function to be monotonically decreasing in the range
    λ∊]0.1, 10[. With parametrization

    (a, b) = (1/4, -6)

    the minimum lies at λ = -b/(2a) = 12 and the lower root at
    λ_r = -(b+d)/(2a) = 12 - d/(2a).

    Solving λ_r = 0.8 for c;

    d = ⎷(b²-4ac) = 2a (12 - 0.8) = 5.6

    c = b² - 5.6² = 36 - 31.36 = 4.64
    """
    a = 1 / 4.
    b = -6.
    c = 4.64
    return a * lambd**2. + b * lambd + c


@pytest.mark.response
def test_find_root():

    def is_converged(value):
        return abs(value) < 1e-7

    myroot = find_root(parabolic_function, is_converged)
    assert myroot == pytest.approx(0.8)
