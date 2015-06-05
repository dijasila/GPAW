import numpy as np

from itertools import product

from scipy.spatial import Delaunay, Voronoi

from gpaw import GPAW


def generate_convex_coeff(NK, N=1):
    a_n = np.linspace(0, 1.0, N + 2)
    gen = product(*([a_n] * NK))

    for a_k in gen:
        a_k = np.array(a_k)
        if np.abs(a_k.sum() - 1) < 1e-10:
            yield a_k


def refine_simplex(simplex_pv, N=1):
    # Refine simplex
    assert len(simplex_pv) < 5
    
    newpoints_pv = []
    n = len(simplex_pv)
    for a_p in generate_convex_coeff(n, N):
        p_v = np.dot(a_p, simplex_pv)
        newpoints_pv.append(p_v.tolist())
    
    return newpoints_pv


def refine_grid(input_pv, N):
    tri = Delaunay(input_pv)
    points_pv = tri.points
    newpoints_pv = []

    for simplex in tri.simplices:
        simplex_pv = points_pv[simplex]
        rpoints_pv = refine_simplex(simplex_pv, N)
        if len(newpoints_pv):
            newpoints_pv.extend(rpoints_pv)
        else:
            newpoints_pv = rpoints_pv

    if not len(newpoints_pv):
        return input_pv

    rpoints_pv = np.append(points_pv,
                           np.array(newpoints_pv), axis=0)

    return unique_rows(rpoints_pv, tol=1e-7)


def tesselate_brillouin_zone(calc, N=5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc)
    bzk_kc, ibzk_kc, bzedges_lkc, bzfaces_fkc = get_BZ(calc)
    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    ibzk_kv = refine_grid(ibzk_kv, N)
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    return ibzk_kc


def unique_rows(ain, tol=1e-7):
    # Round
    a = ain.copy().round((-np.log10(tol).astype(int)))
    order = np.lexsort(a.T)
    a = a[order].copy()
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(1)
    return a[ui]


def get_smallest_Gvecs(cell_cv, n=5):
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2
    G_xv = np.dot(N_xc, B_cv)

    return G_xv


def get_IBZ_vertices(cell_cv, origin_c=None, U_scc=None,
                     time_reversal=None, tol=1e-7):
    if origin_c is None:
        origin_c = np.array([0, 0, 0], float)

    if U_scc is None:
        U_scc = np.array([np.eye(3)])

    inv_cc = -np.eye(3, dtype=int)
    has_inversion = (U_scc == inv_cc).all(2).all(1).any()

    if has_inversion:
        time_reversal = False

    if time_reversal:
        Utmp_scc = np.concatenate([U_scc, -U_scc])
    else:
        Utmp_scc = U_scc

    icell_cv = np.linalg.inv(cell_cv).T
    B_cv = icell_cv * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T

    # Map a random point around
    point_sc = np.dot(origin_c, Utmp_scc.transpose((0, 2, 1)))
    assert len(point_sc) == len(unique_rows(point_sc))
    
    point_sv = np.dot(point_sc, B_cv)
    
    # Translate the points
    n = 5
    G_xv = get_smallest_Gvecs(cell_cv, n=n)
    G_xv = np.delete(G_xv, n**3 // 2, axis=0)

    # Mirror points in plane
    N_xv = G_xv / (((G_xv**2).sum(1))**0.5)[:, np.newaxis]

    tp_sxv = (point_sv[:, np.newaxis] - G_xv[np.newaxis] / 2.)
    delta_sxv = ((tp_sxv * N_xv[np.newaxis]).sum(2)[..., np.newaxis]
                 * N_xv[np.newaxis])
    points_xv = (point_sv[:, np.newaxis] - 2 * delta_sxv).reshape((-1, 3))
    points_xv = np.concatenate([point_sv, points_xv])
    voronoi = Voronoi(points_xv)
    ibzregions = voronoi.point_region[0:len(point_sv)]

    ibzregion = ibzregions[0]
    ibzk_kv = voronoi.vertices[voronoi.regions[ibzregion]]
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    return ibzk_kc


def get_BZ(calc):
    if isinstance(calc, str):
        calc = GPAW(calc)
    symmetry = calc.wfs.kd.symmetry
    cell_cv = calc.wfs.gd.cell_cv
    get_IBZ_vertices(cell_cv)
    
    # The BZ is just the IBZ
    # without symmetries
    bzk_kc = get_IBZ_vertices(cell_cv, tol=1e-7)

    # Use the symmetries to find
    # the IBZ
    ibzk_kc = get_IBZ_vertices(cell_cv,
                               origin_c=np.array([2.9, 2, 1.1]) * 1e-3,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    return bzk_kc, ibzk_kc


