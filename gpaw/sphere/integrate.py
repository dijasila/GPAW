import numpy as np

from gpaw.sphere.lebedev import weight_n


def integrate_lebedev(f_nx):
    """Integrate the function f(r) on the angular Lebedev quadrature.

    Here, n is the quadrature index for the angular dependence of the function
    defined on a spherical grid, while x are some arbitrary extra dimensions.
    """
    return 4. * np.pi * np.tensordot(weight_n, f_nx, axes=([0], [0]))


def integrate_radial_grid(f_gx, r_g, rcut=None):
    """Integrate the function f(r) on the radial grid.

    Computes the integral

    /
    | r^2 dr f(r)
    /

    for the range of values r on the grid r_g (up to rcut, if specified).
    """
    if rcut is not None:
        f_gx, r_g = truncate_radial_grid(f_gx, r_g, rcut)

    # Perform actual integration using the radial trapezoidal rule
    f_xg = np.moveaxis(f_gx, 0, -1)
    f_x = radial_trapz(f_xg, r_g)

    return f_x


def truncate_radial_grid(f_gx, r_g, rcut):
    """Truncate the radial grid representation of a function f(r) at r=rcut.

    If rcut is not part of the original grid, it will be added as a grid point,
    with f(rcut) determined by linear interpolation."""
    assert rcut > 0.
    assert np.any(r_g >= rcut)
    if rcut not in r_g:
        # Find the two points closest to rcut and interpolate between them
        # to get the value at rcut
        g1, g2 = find_two_closest_grid_points(r_g, rcut)
        r1 = r_g[g1]
        r2 = r_g[g2]
        lambd = (rcut - r1) / (r2 - r1)
        f_interpolated_x = (1 - lambd) * f_gx[g1] + lambd * f_gx[g2]
        # Add rcut as a grid point
        r_g = np.append(r_g, np.array([rcut]))
        f_gx = np.append(f_gx, np.array([f_interpolated_x]), axis=0)
    # Pick out the grid points inside rcut
    mask_g = r_g <= rcut
    r_g = r_g[mask_g]
    f_gx = f_gx[mask_g]

    return f_gx, r_g


def find_two_closest_grid_points(r_g, rcut):
    """Find the two closest grid point to a specified rcut."""
    # Find the two smallest absolute differences
    abs_diff_g = abs(r_g - rcut)
    ad1, ad2 = np.partition(abs_diff_g, 1)[:2]

    # Identify the corresponding indices
    g1 = np.where(abs_diff_g == ad1)[0][0]
    g2 = np.where(abs_diff_g == ad2)[0][0]

    return g1, g2


def radial_trapz(f_xg, r_g):
    r"""Integrate the function f(r) using the radial trapezoidal rule.

    Linearly interpolating,

                    r - r0
    f(r) ≃ f(r0) + ‾‾‾‾‾‾‾ (f(r1) - f(r0))      for r0 <= r <= r1
                   r1 - r0

    the integral

    /
    | r^2 dr f(r)
    /

    can be constructed in a piecewise manner from each discretized interval
    r_(n-1) <= r <= r_n, using:

    r1
    /               1
    | r^2 dr f(r) ≃ ‾ (r1^3 f(r1) - r0^3 f(r0))
    /               4
    r0                r1^3 - r0^3
                    + ‾‾‾‾‾‾‾‾‾‾‾ (r1 f(r0) - r0 f(r1))
                      12(r1 - r0)
    """
    assert np.all(r_g >= 0.)
    assert f_xg.shape[-1] == len(r_g)

    # Start and end of each discretized interval
    r0_g = r_g[:-1]
    r1_g = r_g[1:]
    f0_xg = f_xg[..., :-1]
    f1_xg = f_xg[..., 1:]
    assert np.all(r1_g - r0_g > 0.),\
        'Please give the radial grid in ascending order'

    # Linearly interpolate f(r) between r0 and r1 and integrate r^2 f(r)
    # in this area
    integrand_xg = (r1_g**3. * f1_xg - r0_g**3. * f0_xg) / 4.
    integrand_xg += (r1_g**3. - r0_g**3.) * (r1_g * f0_xg - r0_g * f1_xg)\
        / (12. * (r1_g - r0_g))

    # Sum over the discretized integration intervals
    return np.sum(integrand_xg, axis=-1)
