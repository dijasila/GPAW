import numpy as np

from scipy.optimize import minimize

from gpaw.spline import Spline
from gpaw.lfc import LocalizedFunctionsCollection
from gpaw.sphere.lebedev import weight_n


def integrate_lebedev(f_nx):
    """Integrate the function f(r) on the angular Lebedev quadrature.

    Here, n is the quadrature index for the angular dependence of the function
    defined on a spherical grid, while x are some arbitrary extra dimensions.
    """
    return 4. * np.pi * np.tensordot(weight_n, f_nx, axes=([0], [0]))


def integrate_radial_grid(f_xg, r_g, rcut=None):
    """Integrate the function f(r) on the radial grid.

    Computes the integral

    /
    | r^2 dr f(r)
    /

    for the range of values r on the grid r_g (up to rcut, if specified).
    """
    if rcut is not None:
        f_xg, r_g = truncate_radial_grid(f_xg, r_g, rcut)

    # Perform actual integration using the radial trapezoidal rule
    f_x = radial_trapz(f_xg, r_g)

    return f_x


def truncate_radial_grid(f_xg, r_g, rcut):
    """Truncate the radial grid representation of a function f(r) at r=rcut.

    If rcut is not part of the original grid, it will be added as a grid point,
    with f(rcut) determined by linear interpolation."""
    assert rcut > 0.
    assert np.any(r_g >= rcut)
    if rcut not in r_g:
        f_gx = np.moveaxis(f_xg, -1, 0)
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
        f_xg = np.moveaxis(f_gx, 0, -1)
    # Pick out the grid points inside rcut
    mask_g = r_g <= rcut
    r_g = r_g[mask_g]
    f_xg = f_xg[..., mask_g]

    return f_xg, r_g


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


def radial_truncation_function(r_g, rcut, drcut=None, lambd=None):
    r"""Generate smooth radial truncation function θ(r<rc).

    The function is generated to interpolate smoothly between the values

           ( 1    for r <= rc - Δrc/2
    θ(r) = < λ    for r = rc
           ( 0    for r >= rc + Δrc/2

    In the interpolation region, rc - Δrc/2 < r < rc + Δrc/2, the nonanalytic
    smooth function

           ( exp(-1/x)  for x > 0
    f(x) = <
           ( 0          for x <= 0

    is used to define θ(r), in order for all derivatives to be continous:

                        f(1/2-[r-rc]/Δrc)
    θ(r) = ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
           f(1/2-[r-rc]/Δrc) + (1-λ)f(1/2+[r-rc]/Δrc)/λ

    Unless given as an input, we choose 0 < λ < 1 to conserve the spherical
    integration volume, 4π rc^3/3.
    """
    assert np.all(r_g >= 0.)
    if drcut is None:
        # As a default, define Δrc to match twice the grid sampling around the
        # cutoff
        g1, g2 = find_two_closest_grid_points(r_g, rcut)
        drcut = 2 * abs(r_g[g2] - r_g[g1])
    assert rcut > 0. and drcut > 0. and rcut - drcut / 2. >= 0.
    if lambd is None:
        lambd = find_volume_conserving_lambd(rcut, drcut, r_g)
    assert 0. < lambd and lambd < 1.
    assert np.any(r_g >= rcut + drcut / 2.)

    def f(x):
        out = np.zeros_like(x)
        out[x > 0] = np.exp(-1 / x[x > 0])
        return out

    # Create array of ones inside rc + Δrc/2
    theta_g = np.ones_like(r_g)
    theta_g[r_g >= rcut + drcut / 2.] = 0.

    # Add smooth truncation
    gmask = np.logical_and(rcut - drcut / 2. < r_g, r_g < rcut + drcut / 2.)
    tr_g = r_g[gmask]
    theta_g[gmask] = f(1 / 2. - (tr_g - rcut) / drcut) \
        / (f(1 / 2. - (tr_g - rcut) / drcut)
           + (1 - lambd) * f(1 / 2. + (tr_g - rcut) / drcut) / lambd)

    return theta_g


def find_volume_conserving_lambd(rcut, drcut, r_g=None):
    r"""Determine the scaling factor λ to conserve the spherical volume.

    For a given rc and drc, λ is determined to make θ(r) numerically satisfy:
       ∞
       /
    4π | r^2 dr θ(r) = 4π rc^3/3
       /
       0
    """
    if r_g is None:
        r_g = _uniform_radial_grid(rcut, drcut)
    ref = 4 * np.pi * rcut**3. / 3.

    def integration_volume_error(lambd):
        theta_g = radial_truncation_function(r_g, rcut, drcut, lambd)
        vol = 4 * np.pi * radial_trapz(theta_g, r_g)
        return abs(vol - ref)

    opt_result = minimize(integration_volume_error,
                          1 / 2.,  # start guess
                          bounds=[(1e-8, 1 - 1e-8)],
                          tol=1e-8 + 1e-6 * ref)
    if opt_result.success:
        lambd = opt_result.x[0]
    else:
        raise Exception('Could not find an appropriate truncation scaling λ',
                        opt_result.message)

    return lambd


def spherical_truncation_function(gd, spos_c, rcut, drcut=None, lambd=None):
    r"""Generate truncation function θ(r<rc) for a sphere centered at spos_c.

    The smooth radial truncation function θ(r<rc) is used to generate a
    smoothly truncated sphere of radius rcut and centered at the scaled
    position spos_c on the real-space grid described by gd.

    See radial_truncation_function() for the functional form of θ(r<rc).
    """
    if drcut is None:
        drcut = default_spherical_drcut(gd)
    if lambd is None:
        lambd = find_volume_conserving_lambd(rcut, drcut)

    # Lay out truncation function on a radial grid and generate spline
    r_g = _uniform_radial_grid(rcut, drcut)
    theta_g = radial_truncation_function(r_g, rcut, drcut, lambd)
    spline = Spline(l=0, rmax=max(r_g),
                    # The input f_g is the expansion coefficient of the
                    # requested spherical harmonic for the function in
                    # question. For l=0, Y = 1/sqrt(4π):
                    f_g=np.sqrt(4 * np.pi) * theta_g)

    # Evaluate truncation function on the real-space grid
    lfc = LocalizedFunctionsCollection(gd, [[spline]], dtype=float)
    lfc.set_positions([spos_c])
    theta_R = gd.zeros(dtype=float)
    lfc.add(theta_R)

    return theta_R


def default_spherical_drcut(gd):
    """Define default width for the spherical truncation function."""
    # Find "diameter" of each grid point volume
    diam = 2. * (3. * gd.dv / (4. * np.pi))**(1. / 3.)
    # Truncate the sphere smoothly over two such diameters
    drcut = 2 * diam
    return drcut


def _uniform_radial_grid(rcut, drcut):
    return np.linspace(0., rcut + 2 * drcut, 251)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    r_g = np.linspace(0., 4., 200)

    plt.subplot(1, 2, 1)
    plt.plot(r_g, radial_truncation_function(r_g, 1.0, 1.0))
    plt.plot(r_g, radial_truncation_function(r_g, 2.0, 1.0))
    plt.plot(r_g, radial_truncation_function(r_g, 3.0, 1.0))
    plt.subplot(1, 2, 2)
    plt.plot(r_g, radial_truncation_function(r_g, 1.0, 2.0))
    plt.plot(r_g, radial_truncation_function(r_g, 2.0, 2.0))
    plt.plot(r_g, radial_truncation_function(r_g, 3.0, 2.0))
    plt.show()
