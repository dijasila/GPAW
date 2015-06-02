import numpy as np

from itertools import product

from scipy.spatial import Delaunay

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


def inbz(G_xv, k_v, tol=1e-5):
    cond_x = ((G_xv**2).sum(1) / 2 + tol >=
              np.dot(G_xv, k_v))

    return cond_x.all()


def unique_rows(ain, tol=1e-10):
    # Round
    a = ain.copy().round((-np.log10(tol).astype(int)))
    order = np.lexsort(a.T)
    a = a[order].copy()
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(1)
    return a[ui]


def get_smallest_Gvecs(cell_cv):
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    n = 7
    N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2
    G_xv = np.dot(N_xc, B_cv)
    zeropos = n**3 // 2
    G_xv = np.delete(G_xv, zeropos, axis=0)

    i = 0
    while i < len(G_xv):
        G_v = G_xv[i]
        if not inbz(G_xv, G_v / 2):
            G_xv = np.delete(G_xv, i, axis=0)
            i -= 1
        i += 1

    return G_xv


def get_BZ_vertices(cell_cv, tol=1e-10):
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    A_cv = np.linalg.inv(B_cv).T
    G_xv = get_smallest_Gvecs(cell_cv)

    bzk_kc = []
    bzedges_lkc = []
    bzfaces_fkc = []

    # Find all vertices, edges
    # and faces of BZ
    ones_c = np.ones(3)

    for i, G1_v in enumerate(G_xv):
        new_face_vertices = 0
        for j, G2_v in enumerate(G_xv):
            new_vertices = 0
            for k, G3_v in enumerate(G_xv):
                G_cv = np.array([G1_v, G2_v, G3_v])
                G2_c = (G_cv**2).sum(1)
                G_cv *= (2 / G2_c)[:, np.newaxis]

                if np.abs(np.linalg.det(G_cv)) > 1e-10:
                    # Find vertices
                    iG_vc = np.linalg.inv(G_cv)
                    k_v = np.dot(iG_vc, ones_c)
                    k_c = np.dot(A_cv, k_v)

                    # Test if point is in BZ
                    if inbz(G_xv, k_v):
                        bzk_kc.append(k_c.tolist())
                        new_vertices += 1

            # All vertices for G1, G2
            # have been found
            new_face_vertices += new_vertices
            if new_vertices > 1:
                # Order with respect to
                # reduced coordinates
                k_kc = np.array(bzk_kc[-new_vertices:]).copy()
                k_kc = unique_rows(k_kc, tol=tol)
                if len(k_kc) < 2:
                    continue

                for i in [2, 1, 0]:
                    delta = k_kc[0, i] - k_kc[1, i]
                    if np.abs(delta) > 1e-10:
                        sign = np.sign(delta)
                        if sign == -1:
                            k_kc[[0, 1]] = k_kc[[1, 0]]
                        break

                bzedges_lkc.append(k_kc.tolist())

        if new_face_vertices > 2:
            # For plotting purposes
            # and finding unique_rows faces
            # it is important that
            # the face kpoints are ordered
            k_kc = unique_rows(np.array(bzk_kc[-new_face_vertices:]),
                               tol=tol)
            k_kv = np.dot(k_kc, B_cv)

            if len(k_kv) < 3:
                continue

            ktmp_kv = k_kv.copy()

            # Find the starting vertex
            for i in [2, 1, 0]:
                i_v = np.argmax(ktmp_kv, axis=0)
                kmax = ktmp_kv[i_v[i], i]
                i_k = np.nonzero(np.abs(ktmp_kv[:, i] - kmax) < 1e-10)[0]

                if len(i_k) == 1:
                    k0_v = ktmp_kv[i_k[0]]
                    break
                else:
                    ktmp_kv = ktmp_kv[i_k]

            ktmp_kv = k_kv.copy()
            ktmp_kv[0] = k0_v
            nk = len(k_kv)
            for i1 in range(1, nk):
                e_kv = k_kv - ktmp_kv[i1 - 1]

                # Orientation of path
                for ik in range(nk):
                    if np.allclose(e_kv[ik], 0.0):
                        continue

                    ndotv_k = np.dot(np.cross(e_kv[ik], e_kv),
                                     ktmp_kv[0])
                    sign_k = np.sign(ndotv_k)
                    if (sign_k >= 0).all():
                        ktmp_kv[i1] = k_kv[ik]
                        break

                    assert ik != nk - 1

            k_kc = np.dot(ktmp_kv, A_cv.T)

            bzfaces_fkc.append(k_kc.tolist())

    # Unique vertices
    bzk_kc = unique_rows(np.array(bzk_kc), tol=tol).tolist()

    # Unique edges
    bzedges_lK = np.array(bzedges_lkc).reshape(-1, 6)
    bzedges_lkc = unique_rows(bzedges_lK,
                              tol=tol).reshape(-1, 2, 3).tolist()

    return bzk_kc, bzedges_lkc, bzfaces_fkc


def get_IBZ_vertices(cell_cv, bzk_kc, bzedges_lkc,
                     bzfaces_fkc, U_scc, time_reversal,
                     tol=1e-12):

    rint = (-np.log10(tol).astype(int))
    inv_cc = -np.eye(3, dtype=int)
    has_inversion = (U_scc == inv_cc).all(2).all(1).any()

    if has_inversion:
        time_reversal = False

    bzedges_lkc = np.array(bzedges_lkc).tolist()

    bzk_kc = np.array(bzk_kc).tolist()
    icell_cv = np.linalg.inv(cell_cv).T
    B_cv = icell_cv * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T

    G_xv = get_smallest_Gvecs(cell_cv)

    if time_reversal:
        Utmp_scc = np.concatenate([U_scc, -U_scc])
    else:
        Utmp_scc = U_scc

    for U_cc in Utmp_scc:
        MT_vv = np.dot(icell_cv.T, np.dot(U_cc, cell_cv))
        # Find invariant directions
        eig_w, vec_vw = np.linalg.eig(MT_vv)

        zeroeigs_w = np.abs(eig_w - 1.0) < 1e-5

        inds_w = np.nonzero(zeroeigs_w)[0]
        count = np.count_nonzero(zeroeigs_w)

        if count == 2:
            # Invariant plane
            inv_vw = vec_vw[:, inds_w].real

            # Calculate intersection with edges
            for line_kc in bzedges_lkc:
                n_v = np.cross(inv_vw[:, 0], inv_vw[:, 1])

                line_kv = np.dot(line_kc, B_cv)
                k0_v = line_kv[0]
                k1_v = line_kv[1]
                vec_v = k1_v - k0_v

                ndotv = np.dot(n_v, vec_v)

                if np.abs(ndotv) > 1e-5:
                    ki_v = (k0_v - np.dot(n_v, k0_v) / ndotv * vec_v)
                    if inbz(G_xv, ki_v):
                        k_c = np.dot(A_cv, ki_v)
                        bzk_kc.append(k_c.tolist())
        elif count == 1:
            # Invariant line
            inv_v = vec_vw[:, inds_w][:, 0].real

            # Calculate intersection with edges
            for face_kc in bzfaces_fkc:
                face_kv = np.dot(face_kc, B_cv)
                k0_v = face_kv[0]
                k1_v = face_kv[1]
                k2_v = face_kv[2]

                n_v = np.cross(k1_v - k0_v, k2_v - k0_v)
                ndotv = np.dot(n_v, inv_v)

                if np.abs(ndotv) > 1e-10:
                    ki_v = (np.dot(n_v, face_kv[0]) / ndotv * inv_v)
                    if inbz(G_xv, ki_v, tol=1e-3):
                        k_c = np.dot(A_cv, ki_v)
                        bzk_kc.append(k_c.tolist())
        elif count == 0:
            bzk_kc.append([.0, .0, .0])

    bzk_kc = unique_rows(np.array(bzk_kc), tol=tol)

    from gpaw.symmetry import map_k_points
    bz2bz_ks = map_k_points(bzk_kc * .99, U_scc, time_reversal, tol=1e-5)

    assert (bz2bz_ks != -1).all()

    nbzkpts = len(bzk_kc)
    bz2bz_k = -np.ones(nbzkpts + 1, int)
    ibz2bz_k = []
    for k in range(nbzkpts - 1):
        # Reverse order looks more natural
        if bz2bz_k[k] == -1:
            b2b_k = bz2bz_ks[k]
            kstar_kc = unique_rows(bzk_kc[b2b_k], tol=tol)
            kstar_kv = np.dot(kstar_kc, B_cv)

            # Find the point closest to the
            # first octant
            for v in range(3):
                kval = kstar_kv[:, v].max()
                kstar_kv = kstar_kv[kstar_kv[:, v] == kval]

            assert len(kstar_kv) == 1

            kmax_v = kstar_kv[0]
            kmax_c = np.dot(A_cv, kmax_v).round(rint)
            ik = np.argwhere((bzk_kc.round(rint) == kmax_c).all(1))[0][0]

            bz2bz_k[b2b_k] = ik
            ibz2bz_k.append(ik)

    ibzk_kc = unique_rows(bzk_kc[ibz2bz_k]).tolist()
    
    # Map kpoints out
    bzk_kc = np.dot(ibzk_kc, U_scc).reshape((-1, 3))
    bzk_kc = unique_rows(bzk_kc, tol=tol)
    
    return bzk_kc, ibzk_kc


def get_lattice_BZ(cell_cv):
    symmetry = Symmetry([0], cell_cv)
    symmetry.find_lattice_symmetry()
    bzk_kc, bzedges_lkc, bzfaces_fkc = get_BZ_vertices(cell_cv)
    bzk_kc, ibzk_kc = get_IBZ_vertices(cell_cv, bzk_kc,
                                       bzedges_lkc,
                                       bzfaces_fkc,
                                       symmetry.op_scc,
                                       symmetry.time_reversal)

    return bzk_kc, ibzk_kc, bzedges_lkc, bzfaces_fkc


def get_BZ(calc):
    if isinstance(calc, str):
        calc = GPAW(calc)
    symmetry = calc.wfs.kd.symmetry
    cell_cv = calc.wfs.gd.cell_cv
    bzk_kc, bzedges_lkc, bzfaces_fkc = get_BZ_vertices(cell_cv)
    bzk_kc, ibzk_kc = get_IBZ_vertices(cell_cv, bzk_kc,
                                       bzedges_lkc,
                                       bzfaces_fkc,
                                       symmetry.op_scc,
                                       symmetry.time_reversal)

    return bzk_kc, ibzk_kc, bzedges_lkc, bzfaces_fkc
