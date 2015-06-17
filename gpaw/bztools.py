from __future__ import division, print_function

import numpy as np

from scipy.spatial import Delaunay, Voronoi, ConvexHull

from ase.units import Bohr

from gpaw import GPAW
from gpaw.symmetry import Symmetry


def get_lattice_symmetry(cell_cv):
    latsym = Symmetry([0], cell_cv)
    latsym.find_lattice_symmetry()
    return latsym


def tesselate_brillouin_zone(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    latsym = get_lattice_symmetry(cell_cv)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)
    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=latsym.op_scc,
                               time_reversal=latsym.time_reversal)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    refining = True
    while refining:
        refining = False
        tess = Delaunay(ibzk_kv)
        newpoints_pv = []
        for simplex in tess.simplices:
            k_kv = tess.points[simplex]
            for k1_v in k_kv:
                for k2_v in k_kv:
                    dist = ((k1_v - k2_v)**2).sum()**0.5
                    if dist > 1 / density:
                        p_v = (k1_v + k2_v) / 2
                        newpoints_pv.append(p_v)
                        refining = True

        if len(newpoints_pv):
            ibzk_kv = np.append(ibzk_kv, newpoints_pv, axis=0)
        ibzk_kv = unique_rows(ibzk_kv, tol=1e-6)

    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    # Fold out to crystal IBZ
    ibzk_kc = expand_ibz(lU_scc, U_scc, ibzk_kc)

    # Find full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-6)

    # Remove BZ points which are equivalent
    hull = ConvexHull(bzk_kc, qhull_options='Qc')

    vertices = hull.vertices
    coplanar = hull.coplanar

    boundarypoints = np.sort(np.concatenate([vertices, coplanar[:, 0]]))
    bk_kc = hull.points[boundarypoints]

    i = 0
    while i < len(bk_kc):
        dk_kc = bk_kc[i + 1:] - bk_kc[i]
        diff_k = np.abs(dk_kc - dk_kc.round()).sum(1)
        inds_k = np.nonzero(diff_k < 1e-7)[0]
        if len(inds_k):
            bk_kc = np.delete(bk_kc, inds_k + i + 1, axis=0)
        i += 1

    bzk_kc = np.delete(hull.points, boundarypoints, axis=0)
    bzk_kc = np.concatenate([bzk_kc, bk_kc])

    return bzk_kc


def unique_rows(ain, tol=1e-10):
    a = ain.copy().round((-np.log10(tol).astype(int)))
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(1)
    return ain[order][ui]


def get_smallest_Gvecs(cell_cv, n=5):
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2
    G_xv = np.dot(N_xc, B_cv)

    return G_xv


def get_symmetry_operations(U_scc, time_reversal):
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
        
    return Utmp_scc


def get_IBZ_vertices(cell_cv, U_scc=None,
                     time_reversal=None, tol=1e-7):
    # Choose an origin
    origin_c = np.array([0.12, 0.22, 0.21], float)

    if U_scc is None:
        U_scc = np.array([np.eye(3)])

    if time_reversal is None:
        time_reversal = False

    Utmp_scc = get_symmetry_operations(U_scc, time_reversal)

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
        calc = GPAW(calc, txt=None)
    cell_cv = calc.wfs.gd.cell_cv
    
    # Crystal symmetries
    symmetry = calc.wfs.kd.symmetry
    cU_scc = get_symmetry_operations(symmetry.op_scc,
                                     symmetry.time_reversal)

    return get_reduced_BZ(cell_cv, cU_scc, False)


def get_reduced_BZ(cell_cv, cU_scc, time_reversal):
    # Lattice symmetries
    latsym = get_lattice_symmetry(cell_cv)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)
    
    # Find Lattice IBZ
    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=latsym.op_scc,
                               time_reversal=latsym.time_reversal)

    # Expand lattice IBZ to crystal IBZ
    ibzk_kc = expand_ibz(lU_scc, cU_scc, ibzk_kc)

    # Fold out to full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               cU_scc.transpose(0, 2, 1))))

    return bzk_kc, ibzk_kc


def expand_ibz(lU_scc, cU_scc, ibzk_kc):
    # Find right cosets
    cosets = []
    Utmp_scc = lU_scc.copy()
    while len(Utmp_scc):
        U1_cc = Utmp_scc[0]
        Utmp_scc = np.delete(Utmp_scc, 0, axis=0)
        j = 0
        new_coset = [U1_cc]
        while j < len(Utmp_scc):
            U2_cc = Utmp_scc[j]
            U3_cc = np.dot(U1_cc, np.linalg.inv(U2_cc))
            if (U3_cc == cU_scc).all(2).all(1).any():
                new_coset.append(U2_cc)
                Utmp_scc = np.delete(Utmp_scc, j, axis=0)
                j -= 1
            j += 1
        cosets.append(new_coset)

    nibzk_ikc = []
    for U_scc in cosets:
        ibzk_ksc = np.dot(ibzk_kc, np.array(U_scc).transpose((0, 2, 1)))
        dist_ks = ((ibzk_ksc -
                    ibzk_kc[:, np.newaxis])**2).sum(-1)**0.5
        dist_s = dist_ks.sum(0)
        s = np.argmin(dist_s)
        U_cc = U_scc[s]
        nibzk_ikc.append(np.dot(ibzk_kc, U_cc.T))

    ibzk_kc = unique_rows(np.concatenate(nibzk_ikc))

    return ibzk_kc
