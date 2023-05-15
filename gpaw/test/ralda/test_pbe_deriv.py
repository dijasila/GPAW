import pytest
import numpy as np
from gpaw.xc.fxc_kernels import get_pbe_fxc_and_intermediate_derivatives


def eps_unif_x(kf_g):
    return -3 * kf_g / (4.0 * np.pi)


# (We test "small" and "large" densities separately)
@pytest.mark.parametrize(
    'n_g', [
        np.linspace(0.001, 0.02, 10000),
        np.linspace(0.02, 0.5, 1000)
    ])
def test_pbe_x_derivs(n_g):
    # Here we test d(fxc(n))/dn.  We create a linspace of n and
    # calculate all the derivatives.  However we put a dummy (constant)
    # value for s2_g (the gradient) which makes the numerics sensitive
    # (I think) for small densities.
    #
    # This test guards against the problem observed in #723.
    dn = n_g[1] - n_g[0]
    # (assuming np.linspace)

    gradn2 = np.ones_like(n_g)
    kf_g = (3. * np.pi**2 * n_g)**(1 / 3.)
    s2_g = gradn2 / (4 * kf_g**2 * n_g**2)

    fxc_g, F_g, Fn_g, Fnn_g = get_pbe_fxc_and_intermediate_derivatives(
        n_g, s2_g)

    def numderiv(y_g):
        return (y_g[1:] - y_g[:-1]) / dn

    E_g = n_g * eps_unif_x(kf_g) * F_g
    d2Edn2_num = numderiv(numderiv(E_g))

    if 1:
        import matplotlib.pyplot as plt
        ax = plt.gca()

        Fn_num_g = numderiv(F_g)
        #Fnn_num_g = numderiv(Fn_g)
        ax.plot(n_g, Fn_g)
        nhalf_g = n_g[:-1] + dn / 2
        ax.plot(nhalf_g, Fn_num_g, ls='--')

        Fnn_num_g = numderiv(Fn_g)
        ax.plot(n_g, Fnn_g)
        #ax.plot(nhalf_g, Fnn_num_g, ls='--')

        Fnn_num2_g = numderiv(numderiv(F_g))
        ax.plot(n_g[1:-1], Fnn_num2_g, ls='--')
        errs = np.abs(Fnn_num2_g - Fnn_g[1:-1])
        ax.plot(n_g[1:-1], errs)

        ax.plot(n_g, fxc_g)
        ax.plot(n_g[1:-1], d2Edn2_num, ls='--')

        plt.show()

    assert fxc_g[1:-1] == pytest.approx(d2Edn2_num, abs=1e-4, rel=1e-4)
    # We could also test the other derivatives (Fn, Fnn)
    # but that's probably not highest priority.
