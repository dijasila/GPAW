from __future__ import division, print_function

import numpy as np

from scipy.spatial import Delaunay, Voronoi, cKDTree, ConvexHull
from scipy.spatial.distance import pdist

from ase.utils.timing import Timer
from ase.units import Bohr
from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw import GPAW, restart
from gpaw.symmetry import Symmetry, aglomerate_points
from gpaw.utilities.progressbar import ProgressBar
from gpaw.kpt_descriptor import to1bz, kpts2sizeandoffsets


def get_lattice_symmetry(cell_cv):
    latsym = Symmetry([0], cell_cv)
    latsym.find_lattice_symmetry()
    return latsym


def monkhorst_pack_high_symmetry(calc, density, lcd):
    if len(lcd) == 1:
        lcd = lcd[0]
        lcd = np.array([lcd, lcd, lcd])

    atoms, calc = restart(calc, txt=None)
    size, offset = kpts2sizeandoffsets(density=density, even=True,
                                       gamma=True, atoms=atoms)

    size = np.ceil(size / lcd) * lcd
    size, offset = kpts2sizeandoffsets(size=size, gamma=True,
                                       atoms=atoms)
    kpts_kc = monkhorst_pack(size) + np.array(offset)

    print('Monkhorst-Pack grid:', size, offset)

    kpts_kc = to1bz(kpts_kc, calc.wfs.gd.cell_cv)

    return kpts_kc


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

    # This could probably be optimized
    density = density / Bohr
    tess = Delaunay(ibzk_kc)
    pb = ProgressBar()
    while True:
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        maxdist = 0
        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_c = tess.points[k]
            for neighbour in neighbours[k < neighbours]:
                k2_c = tess.points[neighbour]
                dist = vectornorm(np.dot(k1_c - k2_c, B_cv))
                if dist * density < 1:
                    continue
                p_c = (k1_c + k2_c) / 2.
                notcoplan = (vectornorm(tess.points[tess.coplanar[:, 0]] - p_c) > 1e-10).all()
                #print(tess.coplanar)
                if notcoplan:
                    if dist > maxdist:
                        maxdist = dist
                    newpoints.append(p_c)

        if len(newpoints):
            pb.update(1. / (density * maxdist))
            newpoints = np.array(newpoints)
            points = np.append(tess.points, newpoints, axis=0)
            # points = unique_rows(points)
            tess = Delaunay(points)
        else:
            break

    pb.finish()

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kc = tess.points[mask]

    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    # Fold out to crystal IBZ
    ibzk_kc = expand_ibz(lU_scc, U_scc, ibzk_kc)

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    return bzk_kc


def tesselate_brillouin_zone_iter(calc, N=5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    # This could probably be optimized
    tess = Delaunay(ibzk_kc)
    for i in range(N):
        print(i)
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_c = tess.points[k]
            for neighbour in neighbours[k < neighbours]:
                k2_c = tess.points[neighbour]
                p_c = (k1_c + k2_c) / 2.
                newpoints.append(p_c)

        newpoints = np.array(newpoints)
        points = np.append(tess.points, newpoints, axis=0)
        points = unique_rows(points)
        tess = Delaunay(points)

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kc = tess.points[mask]

    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    return bzk_kc


def tesselate_brillouin_zone_eigs(calc, density=3.5, delta=None,
                                  tol=1e-5, shift=0):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calcstr = calc
        calc = GPAW(calc, txt=None)

    ef = calc.get_fermi_level()
    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    # This could probably be optimized
    density = density / Bohr
    tess = Delaunay(ibzk_kc)
    temp = 0.0001
    eigs_kn = []
    while True:
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        neark = []
        maxdist = 0
        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_c = tess.points[k]
            for neighbour in neighbours[k < neighbours]:
                k2_c = tess.points[neighbour]
                dist = vectornorm(np.dot(k1_c - k2_c, B_cv))

                if len(eigs_kn):
                    omega1_n = np.diff(eigs_kn[k])
                    omega2_n = np.diff(eigs_kn[neighbour])
                    f1_n = 1. / (1 + np.exp(eigs_kn[k] / temp))
                    f2_n = 1. / (1 + np.exp(eigs_kn[neighbour] / temp))
                    df1_n = np.diff(f1_n)
                    df2_n = np.diff(f2_n)

                    domega_n = 2 * (omega1_n - omega2_n) / (omega1_n + omega2_n)

                    mask = np.logical_or(np.abs(df1_n) > 0.999,
                                         np.abs(df2_n) > 0.999)
                    mask = np.logical_and(mask,
                                          np.abs(omega1_n + omega2_n) / 2 > 1e-3)

                    domega_n = domega_n[mask]
                    cond = (np.abs(domega_n) < delta).all()

                    if cond:
                        continue

                elif dist * density < 1:
                    continue
                p_c = (k1_c + k2_c) / 2.
                notcoplan = (vectornorm(tess.points[tess.coplanar[:, 0]]
                                        - p_c) > 1e-10).all()
                if notcoplan:
                    if dist > maxdist:
                        maxdist = dist
                    newpoints.append(p_c)
                    neark.append([k, neighbour])

        if delta is not None and maxdist < 1 / density and not len(eigs_kn):
            assert len(newpoints) == 0
            delta = delta
            tmpcalc = GPAW(calcstr, mode='lcao', basis='dzp',
                           fixdensity=True, nbands=-1,
                           kpts=tess.points,
                           symmetry='off', txt=None)#,
                           #communicator=mpi.serial_comm)
            tmpcalc.get_potential_energy()
            for k in range(len(tess.points)):
                eigs_kn.append(tmpcalc.get_eigenvalues(kpt=k) + shift - ef)
                
            continue

        if len(newpoints):
            #pb.update(1. / (density * maxdist))
            newpoints = np.array(newpoints)
            if len(eigs_kn):
                newpoints = unique_rows(newpoints)
                # Do simple LCAO calculations for the
                # eigenvalues
                tmpcalc = GPAW(calcstr, mode='lcao',
                               basis='dzp', fixdensity=True,
                               kpts=newpoints, #realspace=False,
                               symmetry='off', txt=None, nbands=-1)#,
                               #communicator=mpi.serial_comm)
            
                tmpcalc.get_potential_energy()

                mask = []
                for k in range(len(newpoints)):
                    eigs_n = (tmpcalc.get_eigenvalues(kpt=k)
                              + shift - ef)
                    #print(neark[k], neark[k][0], neark[k][1])
                    neareigs_kn = np.array([eigs_kn[neark[k][0]],
                                            eigs_kn[neark[k][1]]])
                    if ((eigs_n - np.sum(neareigs_kn, axis=0) / 2)
                        < 0.1).all():
                        continue

                    mask.append(k)
                    eigs_kn.append(eigs_n)
                newpoints = newpoints[mask]
            print(len(newpoints))

            if len(newpoints) == 0:
                break

            points = np.append(tess.points, newpoints, axis=0) 
            tess = Delaunay(points)
        else:
            break

    pb.finish()

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kc = tess.points[mask]

    print('Number of irreducible k-points: %s' % len(ibzk_kc)) 

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    return ibzk_kc


def tesselate_brillouin_zone_eigs2(calc, density0=3.5, density1=None,
                                   delta=None, radius=None,
                                   special_points=None,
                                   tol=1e-5, shift=0):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calcstr = calc
        calc = GPAW(calc, txt=None)

    ef = calc.get_fermi_level()
    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    latsym = get_lattice_symmetry(cell_cv)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)

    density0 = density0 / Bohr

    if special_points is None:
        special_points_pv = np.array([np.inf, np.inf, np.inf],
                                     float)[np.newaxis]
    else:
        special_points_pc = unfold_points(special_points, lU_scc)
        special_points_pv = np.dot(special_points_pc, B_cv)

    if density1 is None:
        density1 = 5 * density0
    else:
        density1 = density1 / Bohr

    if radius is None:
        radius = 0.1
    else:
        radius = radius / Bohr

    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    def densityfunc(k_v, density0=density0, density1=density1,
                    special_points_pv=special_points_pv, radius=radius):
        """ Make kpoint grid with variable density. """
        distance_to_points = vectornorm(special_points_pv - k_v)

        if np.min(distance_to_points) < radius:
            return density1
        else:
            return density0

    # This could probably be optimized
    ibzk_kv = np.dot(ibzk_kc, B_cv)
    tess = Delaunay(ibzk_kv)
    pb = ProgressBar()
    temp = 0.025
    eigs_kn = []
    while True:
        print('iter')
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        maxdist = 0
        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_v = tess.points[k]
            for neighbour in neighbours[k < neighbours]:
                k2_v = tess.points[neighbour]
                densityk1 = densityfunc(k1_v)
                densityk2 = densityfunc(k2_v)
                dist = vectornorm(k1_v - k2_v)

                if len(eigs_kn):
                    f1_n = 1. / (1 + np.exp(eigs_kn[k] / temp))
                    f2_n = 1. / (1 + np.exp(eigs_kn[neighbour] / temp))
                    df1_n = np.diff(f1_n)
                    df2_n = np.diff(f2_n)

                    df_n = np.abs(df1_n - df2_n)
                    cond = (df_n < 0.25).all()

                    if cond:
                        continue
                elif dist * densityk1 < 1 and dist * densityk2 < 1:
                    continue
                p_v = (k1_v + k2_v) / 2.
                notcoplan = (vectornorm(tess.points[tess.coplanar[:, 0]]
                                        - p_v) > 1e-10).all()
                if notcoplan:
                    if dist > maxdist:
                        maxdist = dist
                    newpoints.append(p_v)

        if delta is not None and maxdist < 1 / density1 and not len(eigs_kn):
            assert len(newpoints) == 0
            delta = delta
            
            print('eigs')
            tmpcalc = GPAW(calcstr, mode='lcao', basis='dzp',
                           fixdensity=True, nbands=-1,
                           kpts=np.dot(tess.points, A_cv.T),
                           realspace=False,
                           symmetry='off', txt=None,
                           communicator=mpi.serial_comm)
            tmpcalc.get_potential_energy()
            for k in range(len(tess.points)):
                eigs_kn.append(tmpcalc.get_eigenvalues(kpt=k) + shift - ef)

            continue

        if len(newpoints):
            #pb.update(1. / (density * maxdist))
            newpoints = np.array(newpoints)
            if len(eigs_kn):
                newpoints = unique_rows(newpoints)
                # Do simple LCAO calculations for the
                # eigenvalues
                print(len(newpoints))
                tmpcalc = GPAW(calcstr, mode='lcao',
                               basis='dzp', fixdensity=True,
                               kpts=np.dot(newpoints, A_cv.T),
                               realspace=False,
                               symmetry='off', txt=None, nbands=-1,
                               communicator=mpi.serial_comm)
            
                tmpcalc.get_potential_energy()

                for k in range(len(newpoints)):
                    eigs_kn.append(tmpcalc.get_eigenvalues(kpt=k)
                                   + shift - ef)

            points = np.append(tess.points, newpoints, axis=0) 
            tess = Delaunay(points)
        else:
            break

    pb.finish()

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kv = tess.points[mask]
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    print('Number of irreducible k-points: %s' % len(ibzk_kc)) 

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    return ibzk_kc


def tesselate_brillouin_zone_eigs3(calc, density0=3.5, density1=None,
                                   delta=None, radius=None,
                                   special_points=None,
                                   tol=1e-5, shift=0):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calcstr = calc
        calc = GPAW(calc, txt=None)

    ef = calc.get_fermi_level()
    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    latsym = get_lattice_symmetry(cell_cv)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)

    density0 = density0 / Bohr

    if special_points is None:
        special_points_pv = np.array([np.inf, np.inf, np.inf],
                                     float)[np.newaxis]
    else:
        special_points_pc = unfold_points(special_points, lU_scc)
        special_points_pv = np.dot(special_points_pc, B_cv)

    if density1 is None:
        density1 = 5 * density0
    else:
        density1 = density1 / Bohr

    if radius is None:
        radius = 0.1
    else:
        radius = radius / Bohr

    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    def densityfunc(k_v, density0=density0, density1=density1,
                    special_points_pv=special_points_pv, radius=radius):
        """ Make kpoint grid with variable density. """
        distance_to_points = vectornorm(special_points_pv - k_v)

        if np.min(distance_to_points) < radius:
            return density1
        else:
            return density0

    # This could probably be optimized
    ibzk_kv = np.dot(ibzk_kc, B_cv)
    tess = Delaunay(ibzk_kv)
    pb = ProgressBar()
    temp = 0.025
    eigs_kn = []
    while True:
        print('iter')
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        maxdist = 0
        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_v = tess.points[k]
            for neighbour in neighbours[k < neighbours]:
                k2_v = tess.points[neighbour]
                densityk1 = densityfunc(k1_v)
                densityk2 = densityfunc(k2_v)
                dist = vectornorm(k1_v - k2_v)

                if len(eigs_kn):
                    omega1_n = np.diff(eigs_kn[k])
                    omega2_n = np.diff(eigs_kn[neighbour])
                    f1_n = 1. / (1 + np.exp(eigs_kn[k] / temp))
                    f2_n = 1. / (1 + np.exp(eigs_kn[neighbour] / temp))
                    df1_n = np.diff(f1_n)
                    df2_n = np.diff(f2_n)

                    #integrand1_n = df1_n / omega1_n
                    #integrand2_n = df2_n / omega2_n
                    #dI_n = np.abs(integrand1_n - integrand2_n)
                    #cond = (dI_n < 0.1).all()

                    df_n = np.abs(df1_n - df2_n)
                    domega_n = 2 * (omega1_n - omega2_n) / (omega1_n + omega2_n)
                    cond = ((np.abs(domega_n) < delta).all() or 
                            (np.abs(omega1_n + omega2_n) / 2 < 1e-3).all())

                    if cond:
                        cond = (df_n < 0.5).all()

                    if cond:
                        continue
                elif dist * densityk1 < 1 and dist * densityk2 < 1:
                    continue
                p_v = (k1_v + k2_v) / 2.
                notcoplan = (vectornorm(tess.points[tess.coplanar[:, 0]]
                                        - p_v) > 1e-10).all()
                if notcoplan:
                    if dist > maxdist:
                        maxdist = dist
                    newpoints.append(p_v)

        if delta is not None and maxdist < 1 / density1 and not len(eigs_kn):
            assert len(newpoints) == 0
            delta = delta
            
            print('eigs')
            tmpcalc = GPAW(calcstr, mode='lcao', basis='dzp',
                           fixdensity=True, nbands=-1,
                           kpts=np.dot(tess.points, A_cv.T),
                           realspace=False,
                           symmetry='off', txt=None,
                           communicator=mpi.serial_comm)
            tmpcalc.get_potential_energy()
            for k in range(len(tess.points)):
                eigs_kn.append(tmpcalc.get_eigenvalues(kpt=k) + shift - ef)

            continue

        if len(newpoints):
            #pb.update(1. / (density * maxdist))
            newpoints = np.array(newpoints)
            if len(eigs_kn):
                newpoints = unique_rows(newpoints)
                # Do simple LCAO calculations for the
                # eigenvalues
                print(len(newpoints))
                tmpcalc = GPAW(calcstr, mode='lcao',
                               basis='dzp', fixdensity=True,
                               kpts=np.dot(newpoints, A_cv.T),
                               realspace=False,
                               symmetry='off', txt=None, nbands=-1,
                               communicator=mpi.serial_comm)
            
                tmpcalc.get_potential_energy()

                for k in range(len(newpoints)):
                    eigs_kn.append(tmpcalc.get_eigenvalues(kpt=k)
                                   + shift - ef)

            points = np.append(tess.points, newpoints, axis=0) 
            tess = Delaunay(points)
        else:
            break

    pb.finish()

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kv = tess.points[mask]
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    print('Number of irreducible k-points: %s' % len(ibzk_kc)) 

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    return ibzk_kc


def tesselate_brillouin_zone_variable_density(calc, density0=3.5,
                                              density1=None, radius=None,
                                              special_points=None):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    latsym = get_lattice_symmetry(cell_cv)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)

    density0 = density0 / Bohr

    if special_points is None:
        special_points_pv = np.array([np.inf, np.inf, np.inf],
                                     float)[np.newaxis]
    else:
        special_points_pc = unfold_points(special_points, lU_scc)
        special_points_pv = np.dot(special_points_pc, B_cv)

    if density1 is None:
        density1 = 5 * density0
    else:
        density1 = density1 / Bohr

    if radius is None:
        radius = 0.1
    else:
        radius = radius / Bohr

    #ibzk_kc = get_IBZ_vertices(cell_cv,
    #                           U_scc=latsym.op_scc,
    #                           time_reversal=latsym.time_reversal)

    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    #ibzk_kc = expand_ibz(lU_scc, U_scc, ibzk_kc)

    def densityfunc(k_v, density0=density0, density1=density1,
                    special_points_pv=special_points_pv, radius=radius):
        """ Make kpoint grid with variable density. """
        distance_to_points = vectornorm(special_points_pv - k_v)

        if np.min(distance_to_points) < radius:
            return density1
        else:
            return density0

    # This could probably be optimized
    ibzk_kv = np.dot(ibzk_kc, B_cv)
    tess = Delaunay(ibzk_kv)
    pb = ProgressBar()
    while True:
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        maxdist = 0

        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_v = tess.points[k]
            for neighbour in neighbours[k < neighbours]:
                k2_v = tess.points[neighbour]
                dist = vectornorm(k1_v - k2_v)
                densityk1 = densityfunc(k1_v)
                densityk2 = densityfunc(k2_v)
                if dist * densityk1 < 1 and dist * densityk2 < 1:
                    continue
                p_v = (k1_v + k2_v) / 2.
                notcoplan = (vectornorm(tess.points[tess.coplanar[:, 0]] - p_v) > 1e-10).all()
                if notcoplan:
                    if dist > maxdist:
                        maxdist = dist
                    newpoints.append(p_v)

        if len(newpoints):
            newpoints = np.array(newpoints)
            points = np.append(tess.points, newpoints, axis=0)
            points = unique_rows(points)
            tess = Delaunay(points)
            pb.update(1. / (density1 * maxdist))
        else:
            break
    pb.finish()

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kv = tess.points[mask]
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    print('Number of irreducible k-points: %s' % len(ibzk_kc)) 

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    print('Number of k-points: %s' % len(bzk_kc)) 

    return bzk_kc


def tesselate_brillouin_zone_variable_density2(calc, density0=3.5,
                                               density1=None, radius=None,
                                               special_points=None):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)

    # Tesselate lattice IBZ
    symmetry = calc.wfs.kd.symmetry
    U_scc = get_symmetry_operations(symmetry.op_scc,
                                    symmetry.time_reversal)

    latsym = get_lattice_symmetry(cell_cv)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)

    density0 = density0 / Bohr

    if special_points is None:
        special_points_pv = np.array([np.inf, np.inf, np.inf],
                                     float)[np.newaxis]
    else:
        special_points_pc = unfold_points(special_points, lU_scc)
        special_points_pv = np.dot(special_points_pc, B_cv)

    if density1 is None:
        density1 = 5 * density0
    else:
        density1 = density1 / Bohr

    if radius is None:
        radius = 0.1
    else:
        radius = radius / Bohr

    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=symmetry.op_scc,
                               time_reversal=symmetry.time_reversal)

    def densityfunc(k_v, density0=density0, density1=density1,
                    special_points_pv=special_points_pv, radius=radius):
        """ Make kpoint grid with variable density. """
        distance_to_points = vectornorm(special_points_pv - k_v)

        if np.min(distance_to_points) < radius:
            return density1
        else:
            return density0

    # This could probably be optimized
    #ibzk_kv = np.dot(ibzk_kc, B_cv)
    tess = Delaunay(ibzk_kc)
    pb = ProgressBar()
    while True:
        # Point to simplex
        pts_k = [[] for n in xrange(tess.npoints)]
        for s, K_k in enumerate(tess.simplices):
            k_kc = tess.points[K_k]
            vol = np.abs(np.linalg.det(k_kc[1:] - k_kc[0]))

            if vol < 1e-10:
                continue

            for K in K_k:
                pts_k[K].append(s)

        # Change to numpy arrays:
        for k in xrange(tess.npoints):
            pts_k[k] = np.array(pts_k[k], int)

        neighbours_k = [None for n in xrange(tess.npoints)]
        for k in xrange(tess.npoints):
            neighbours_k[k] = np.unique(tess.simplices[pts_k[k]])

        newpoints = []
        maxdist = 0

        for k in range(tess.npoints):
            neighbours = neighbours_k[k]
            k1_c = tess.points[k]
            k1_v = np.dot(k1_c, B_cv)
            for neighbour in neighbours[k < neighbours]:
                k2_c = tess.points[neighbour]
                k2_v = np.dot(k2_c, B_cv)
                dist = vectornorm(k1_v - k2_v)
                densityk1 = densityfunc(k1_v)
                densityk2 = densityfunc(k2_v)
                if dist * densityk1 < 1 and dist * densityk2 < 1:
                    continue
                p_c = (k1_c + k2_c) / 2.
                notcoplan = (vectornorm(tess.points[tess.coplanar[:, 0]] - p_c) > 1e-10).all()
                if notcoplan:
                    if dist > maxdist:
                        maxdist = dist
                    newpoints.append(p_c)

        if len(newpoints):
            newpoints = np.array(newpoints)
            points = np.append(tess.points, newpoints, axis=0)
            points = unique_rows(points)
            tess = Delaunay(points)
            pb.update(1. / (density1 * maxdist))
        else:
            break
    pb.finish()

    mask = np.ones(len(tess.points), bool)
    mask[tess.coplanar[:, 0]] = False
    ibzk_kc = tess.points[mask]
    print('Number of irreducible k-points: %s' % len(ibzk_kc)) 

    # Find full BZ modulo 1.
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               U_scc.transpose(0, 2, 1))),
                         tol=1e-8, mod=1)

    print('Number of k-points: %s' % len(bzk_kc)) 

    return bzk_kc


def tesselate_brillouin_zone2(calc, density=3.5):
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
    lU_svv = np.dot(B_cv.T, np.dot(lU_scc, A_cv)).transpose((1, 0, 2))
        
    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=latsym.op_scc,
                               time_reversal=latsym.time_reversal)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 3 / (4 * np.pi) * (1. / density)**(1. / 3)

    def displace_points(points_xv, n=3):
        nd = points_xv.shape[1]
        N_xc = np.indices(tuple([n] * nd)).reshape((nd, n**nd)).T - n // 2
        N_xv = np.dot(N_xc, B_cv)
        newpoints_xc = (points_xv[np.newaxis]
                        + N_xv[:, np.newaxis]).reshape(-1, nd)
        return unique_rows(newpoints_xc, tol=1e-8)

    IBZ = Delaunay(ibzk_kv)
    points = unique_rows(np.concatenate(np.dot(IBZ.points,
                                               lU_svv.transpose(0, 2, 1))),
                         tol=1e-8)
    points = displace_points(points)

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    while True:
        print(len(points))
        vor = Voronoi(points)
        maxdist_v = np.zeros(len(vor.vertices), float) + 1e9
        for j, region in enumerate(vor.point_region):
            vertexid = np.array(vor.regions[region])
            vertexid = vertexid[vertexid != -1]
            dist = ((vor.vertices[vertexid] - vor.points[j])**2).sum(-1)**0.5
            maxdist_v[vertexid] = np.min(np.array([maxdist_v[vertexid].copy(),
                                                   dist]), axis=0)

        mask = np.argwhere(IBZ.find_simplex(vor.vertices) >= 0)[:, 0]
        maxdist_v = maxdist_v[mask]
        if np.max(maxdist_v) < nn_dist:
            break

        indices = [np.argsort(maxdist_v)[-1]]
        newpoints = vor.vertices[mask[indices]]
        newpoints = unique_rows(np.concatenate(np.dot(newpoints,
                                                      lU_svv.transpose(0, 2, 1))),
                                tol=1e-8)
        newpoints = displace_points(newpoints)
        points = np.append(points, newpoints, axis=0)

    points = vor.points[IBZ.find_simplex(vor.points) >= 0]
    #points = unique_rows(np.concatenate(np.dot(points,
    #                                           lU_svv.transpose(0, 2, 1))),
    #                     tol=1e-8)

    #s = slice(0, 2)
    #print(lU_svv[s])
    #lU_svv = lU_svv[s].transpose(0, 2 ,1)
    #points = np.concatenate(np.dot(points, lU_svv[s]))
    #points = unique_rows(points, tol=1e-8)

    bzk_kc = np.dot(points, A_cv.T)

    return bzk_kc

def unfold_points(points, U_scc, tol=1e-8, mod=None):
    points = np.concatenate(np.dot(points, U_scc.transpose(0, 2, 1)))
    return unique_rows(points, tol=tol, mod=mod)

def tesselate_brillouin_zone3(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                     sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 3 / (4 * np.pi) * (1. / density)**(1. / 3)

    def displace_points(points_xv, n=3):
        nd = points_xv.shape[1]
        N_xc = np.indices(tuple([n] * nd)).reshape((nd, n**nd)).T - n // 2
        N_xv = np.dot(N_xc, B_cv)
        newpoints_xv = (points_xv[np.newaxis]
                        + N_xv[:, np.newaxis]).reshape(-1, nd)
        return unique_rows(newpoints_xv, tol=1e-8)

    IBZ = Delaunay(ibzk_kv)
    points = unfold_points(IBZ.points, U_svv)
    points = displace_points(points)

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    vor = Voronoi(points, incremental=True)
    timer = Timer()

    pb = ProgressBar()
    while True:
        tree = cKDTree(vor.points)

        with timer('Find nearest neighbour'):
            mindist_v = tree.query(vor.vertices)[0]

        mask = np.argwhere(IBZ.find_simplex(vor.vertices) >= 0)[:, 0]
        mindist_v = mindist_v[mask]
        if np.max(mindist_v) < nn_dist:
            break

        indices = np.argsort(mindist_v)[-1]
        newpoints = unfold_points([vor.vertices[mask[indices]],],
                                  U_svv)
        newpoints = displace_points(newpoints)
        with timer('add_points'):
            # points = np.concatenate([vor.points, newpoints])
            # vor.points = points[:-2]
            # vor.add_points(points[-2:], restart=True)
            vor.add_points(newpoints) # , restart=True)
        pb.update(nn_dist / mindist_v[indices])
    pb.finish()
    
    vor.close()

    timer.write()
    points = vor.points[IBZ.find_simplex(vor.points) >= 0]
    # points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def tesselate_brillouin_zone4(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 3 / (4 * np.pi) * (1. / density)**(1. / 3)

    def displace_points(points_xv, n=3):
        nd = points_xv.shape[1]
        N_xc = np.indices(tuple([n] * nd)).reshape((nd, n**nd)).T - n // 2
        N_xv = np.dot(N_xc, B_cv)
        newpoints_xv = (points_xv[np.newaxis]
                        + N_xv[:, np.newaxis]).reshape(-1, nd)
        return unique_rows(newpoints_xv, tol=1e-8)

    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = unfold_points(IBZ.points, U_svv)
    points = displace_points(points)

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()
    while True:
        with timer('Voronoi'):
            vor = Voronoi(points)
        with timer('KDTree'):
            tree = cKDTree(points)

        radius_v = tree.query(vor.vertices)[0]

        inibz_v = IBZ.find_simplex(vor.vertices) >= 0
        inibz_p = np.argwhere(IBZ.find_simplex(points) >= 0)[:, 0]
        
        # What points are close enough to the
        # IBZ to be relevant
        projectedvertices_pv = project_onto_3dhull(IBZhull, vor.vertices)
        dist_v = vectornorm(projectedvertices_pv - vor.vertices)

        vertexmask_v = dist_v - radius_v <= 1e-8
        nearpoints = inibz_p.tolist()
        for v, vertex in enumerate(vertexmask_v):
            if vertexmask_v[v]:
                nearpoints.extend(tree.query_ball_point(vor.vertices[v],
                                                        radius_v[v] + 1e-8))

        nearpoints = np.unique(np.array(nearpoints))

        inibz_v = np.argwhere(inibz_v)[:, 0]
        radius_v = radius_v[inibz_v]

        if np.max(radius_v) < nn_dist:
            break

        index = np.argsort(radius_v)[-1]
        newpoint = vor.vertices[inibz_v[index]]
        newpoints = unfold_points([newpoint,],
                                  U_svv)
        newpoints = displace_points(newpoints)
        points = np.concatenate([points[nearpoints],
                                 newpoints])
        pb.update(nn_dist / radius_v[index])

    pb.finish()

    timer.write()
    points = vor.points[IBZ.find_simplex(vor.points) >= 0]
    points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc

def tesselate_brillouin_zone5(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 3 / (4 * np.pi) * (1. / density)**(1. / 3)
    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = IBZ.points.copy()

    print('nn_dist', nn_dist)
    print('ibz', IBZ.points)

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()
    while True:
        # First calculate the tesselation
        assert (IBZ.find_simplex(points) >= 0).all()
        tess = Delaunay(points)
        vertices_v = np.zeros((tess.nsimplex, 3), float)
        nndist_v = np.zeros(tess.nsimplex, float)

        for iv, simplex in enumerate(tess.simplices):
            p_pv = tess.points[simplex]
            vol = np.abs(np.linalg.det(p_pv[1:] - p_pv[0]))
            if vol < 1e-10:
                continue
            norm_p = vectornorm(p_pv)
            center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                              norm_p[1:]**2 - norm_p[0]**2) / 2.
            vertices_v[iv] = center_v

            dist_p = pdist(p_pv)
            nndist_v[iv] = np.max(dist_p)

        tree = cKDTree(vertices_v)

        # Determine whether any edges or faces are
        # encroached: This happens when circumcenters
        # are located outside the IBZ. Project these
        # onto the hull of IBZ and treat lines first
        inibz_v = IBZ.find_simplex(vertices_v) >= 0
        notinibz_v = np.logical_not(inibz_v)

        if False: #not inibz_v.all():
            indices = np.argwhere(nndist_v[notinibz_v] > nn_dist)[:, 0]
            if len(indices):
                newpoints = refine_edge_or_face(IBZhull,
                                                vertices_v[indices])
                newpoint = newpoints[0]

                points = np.concatenate([points, [newpoint]])
                assert len(points) == len(unique_rows(points))
                continue

        vertices_v = vertices_v[inibz_v]
        nndist_v = nndist_v[inibz_v]

        if np.max(nndist_v) < nn_dist:
            break

        index = np.argsort(nndist_v)[-1]
        newpoints = [vertices_v[index]]

        # Try to check whether the voronoi
        # decomposition is valid
        #print(newpoints)
        ansatzpoints = np.concatenate([points, newpoints])
        # vor = Voronoi(ansatzpoints)
        tess2 = Delaunay(ansatzpoints)
        vertices2_v = np.zeros((tess2.nsimplex, 3), float)

        for iv, simplex in enumerate(tess2.simplices):
            p_pv = tess2.points[simplex]
            vol = np.abs(np.linalg.det(p_pv[1:] - p_pv[0]))
            if vol < 1e-10:
                continue
            norm_p = vectornorm(p_pv)
            center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                              norm_p[1:]**2 - norm_p[0]**2) / 2.
            vertices2_v[iv] = center_v

        notinibz = IBZ.find_simplex(vertices2_v) < 0
        vertices2_v = vertices2_v[notinibz]
        tree2 = cKDTree(vertices2_v)
        pairs = tree2.query_ball_tree(tree, 1e-8)
        indices = np.argwhere([len(pair) == 0 for pair in pairs])[:, 0]

        if len(indices):
            newpoints = refine_edge_or_face(IBZhull,
                                            vertices2_v[indices])
            projected = project_onto_3dhull(IBZhull, newpoints)
            projected2 = project_onto_3dhull(IBZhull, vertices2_v[indices])
            assert (vectornorm(projected - newpoints) < 1e-8).all(), print(vectornorm(projected - newpoints), newpoints, projected2, vertices2_v[indices])
            newpoints = [newpoints[0]]
            #print('enc', newpoints)

        points = np.concatenate([points, newpoints])
        assert len(points) == len(unique_rows(points))
        pb.update(nn_dist / nndist_v[index])

    print(tess.coplanar)



    pb.finish()
    timer.write()
    #points = vor.points[IBZ.find_simplex(vor.points) >= 0]
#    points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def tesselate_brillouin_zone6(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 1. / density
    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = IBZ.points.copy()

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()
    while True:
        # First calculate the tesselation
        assert (IBZ.find_simplex(points) >= 0).all()
        tess = Delaunay(points)
        vertices_v = np.zeros((tess.nsimplex, 3), float)
        nndist_v = np.zeros(tess.nsimplex, float)

        for iv, simplex in enumerate(tess.simplices):
            p_pv = tess.points[simplex]
            vol = np.abs(np.linalg.det(p_pv[1:] - p_pv[0]))
            if vol < 1e-10:
                continue
            norm_p = vectornorm(p_pv)
            center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                              norm_p[1:]**2 - norm_p[0]**2) / 2.
            vertices_v[iv] = center_v
            dist_p = pdist(p_pv)
            nndist_v[iv] = np.max(dist_p)

        if np.max(nndist_v) < nn_dist:
            print(np.max(nndist_v))
            break

        tree = cKDTree(vertices_v.copy())

        # Determine whether any edges or faces are
        # encroached: This happens when circumcenters
        # are located outside the IBZ. Project these
        # onto the hull of IBZ and treat lines first
        inibz_v = IBZ.find_simplex(vertices_v) >= 0
        notinibz_v = np.logical_not(inibz_v)

        if notinibz_v.any():
            projectedpoints = project_onto_3dhull(IBZhull,
                                                  vertices_v[notinibz_v])
            vertices_v[notinibz_v] = projectedpoints


        index = np.argsort(nndist_v)[-1]
        newpoints = [vertices_v[index]]

        # Try to check whether the voronoi
        # decomposition is valid
        ansatzpoints = np.concatenate([points, newpoints])
        tess2 = Delaunay(ansatzpoints)
        vertices2_v = np.zeros((tess2.nsimplex, 3), float)

        for iv, simplex in enumerate(tess2.simplices):
            p_pv = tess2.points[simplex]
            vol = np.abs(np.linalg.det(p_pv[1:] - p_pv[0]))
            if vol < 1e-10:
                continue
            norm_p = vectornorm(p_pv)
            center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                              norm_p[1:]**2 - norm_p[0]**2) / 2.
            vertices2_v[iv] = center_v

        notinibz = IBZ.find_simplex(vertices2_v) < 0
        if notinibz.any():
            vertices2_v = vertices2_v[notinibz]
            tree2 = cKDTree(vertices2_v)
            pairs = tree2.query_ball_tree(tree, 1e-8)
            indices = np.argwhere([len(pair) == 0 for pair in pairs])[:, 0]

            if len(indices):
                newpoints = refine_edge_or_face(IBZhull,
                                                vertices2_v[indices])
                newpoints = [newpoints[0]]

        points = np.concatenate([points, newpoints])
        assert len(points) == len(unique_rows(points))
        pb.update(nn_dist / nndist_v[index])

    assert len(tess.coplanar) == 0

    pb.finish()
    timer.write()
    points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def tesselate_brillouin_zone7(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 1. / density
    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = IBZ.points.copy()

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()

    from itertools import combinations

    while True:
        # First calculate the tesselation
        assert (IBZ.find_simplex(points) >= 0).all()
        with timer('delaunay'):
            vor = Voronoi(points)
            vertices_v = vor.vertices
        vert2pts_vp = [[] for i in range(len(vertices_v))]
        for point, region in enumerate(vor.point_region):
            verts = [vt for vt in vor.regions[region] if vt != -1]
            for vert in verts:
                vert2pts_vp[vert].append(point)

        nndist_v = np.zeros(len(vor.vertices), float)
        with timer('circum1'):
            for iv, pts in enumerate(vert2pts_vp):
                if (np.abs(vertices_v[iv]) > 1e10).any():
                    continue
                p_pv = vor.points[pts]

                for comb in combinations(np.arange(len(pts)), 4):
                    comb = np.array(comb)
                    vol = np.abs(np.linalg.det(p_pv[comb[1:]] - p_pv[comb[0]]))
                    if vol > 1e-15:
                        break

                if vol < 1e-15:
                    continue

                dist_p = pdist(p_pv)
                nndist_v[iv] = np.max(dist_p)

        if np.max(nndist_v) < nn_dist:
            break

        # Determine whether any edges or faces are
        # encroached: This happens when circumcenters
        # are located outside the IBZ. Project these
        # onto the hull of IBZ and treat lines first
        notinibz_v = IBZ.find_simplex(vertices_v) < 0

        if notinibz_v.any():
            projectedpoints = project_onto_3dhull(IBZhull,
                                                  vertices_v[notinibz_v])
            vertices_v[notinibz_v] = projectedpoints

        index = np.argmax(nndist_v)
        newpoints = [vertices_v[index]]

        # Try to check whether the voronoi
        # decomposition is valid
        ansatzpoints = np.concatenate([points, newpoints])
        vor2 = Voronoi(ansatzpoints)
        vid = np.array(vor2.regions[vor2.point_region[-1]])
        vid = vid[vid != -1]
        vertices2_v = vor2.vertices[vid]
        vertices2_v = vertices2_v[(np.abs(vertices2_v) < 1e10).all(-1)]

        with timer('newpoints'):
            notinibz = IBZ.find_simplex(vertices2_v) < 0
            if notinibz.any():
                vertices2_v = vertices2_v[notinibz]

                newpoints = refine_edge_or_face(IBZhull, vertices2_v)
                newpoints = [newpoints[0]]

        points = np.concatenate([points, newpoints])
        assert len(points) == len(unique_rows(points))
        pb.update(nn_dist / nndist_v[index])
    tess = Delaunay(points)
    assert len(tess.coplanar) == 0

    pb.finish()
    timer.write()
    points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def tesselate_brillouin_zone8(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 1. / density
    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = IBZ.points.copy()

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()

    from scipy.spatial import minkowski_distance as md

    while True:
        # First calculate the tesselation
        assert (IBZ.find_simplex(points) >= 0).all()
        with timer('voronoi1'):
            tess = Delaunay(points)
            
        with timer('Calculate nndist'):
            simplexpoints_spv = points[tess.simplices]
            s_spv = simplexpoints_spv
            nndist_s = np.array([md(s_spv[:, 0], s_spv[:, 1]),
                                 md(s_spv[:, 0], s_spv[:, 2]),
                                 md(s_spv[:, 0], s_spv[:, 3]),
                                 md(s_spv[:, 1], s_spv[:, 2]),
                                 md(s_spv[:, 1], s_spv[:, 3]),
                                 md(s_spv[:, 2], s_spv[:, 3])]).max(0)
            simplexpoints_spv = simplexpoints_spv[:, 1:] - \
                                simplexpoints_spv[:, 0][:, np.newaxis]
            volumes_s = np.abs(np.linalg.det(simplexpoints_spv))
            nndist_s[volumes_s < 1e-10] = 0.0

        if np.max(nndist_s) < nn_dist:
            break

        index = np.argmax(nndist_s)
        
        p_pv = tess.points[tess.simplices[index]]
        norm_p = vectornorm(p_pv)
        center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                          norm_p[1:]**2 - norm_p[0]**2) / 2.

        newpoints = center_v[np.newaxis]
        if IBZ.find_simplex(center_v) < 0:
            newpoints = project_onto_3dhull(IBZhull, newpoints)

        # Try to check whether the voronoi
        # decomposition is valid
        ansatzpoints = np.concatenate([points, newpoints])
        with timer('voronoi2'):
            vor2 = Voronoi(ansatzpoints)
        vid = np.array(vor2.regions[vor2.point_region[-1]])
        vid = vid[vid != -1]
        vertices2_v = vor2.vertices[vid]
        vertices2_v = vertices2_v[(np.abs(vertices2_v) < 1e10).all(-1)]

        notinibz = IBZ.find_simplex(vertices2_v) < 0
        if notinibz.any():
            vertices2_v = vertices2_v[notinibz]
            newpoints = refine_edge_or_face(IBZhull, vertices2_v)
            newpoints = [newpoints[0]]

        points = np.concatenate([points, newpoints])
        assert len(points) == len(unique_rows(points))
        pb.update(nn_dist / nndist_s[index])
    tess = Delaunay(points)
    assert len(tess.coplanar) == 0

    pb.finish()
    timer.write()
    points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def tesselate_brillouin_zone9(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 1. / density
    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = IBZ.points.copy()

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()

    from scipy.spatial import minkowski_distance as md

    import scipy
    print('scipy version: ' + scipy.__version__)

    vor = Voronoi(points.copy(), incremental=True,
                  qhull_options='Qc Qx QJ Q11')
    tess = Delaunay(points.copy(), incremental=True,
                    qhull_options='Qc Qx QJ Q11')

    while True:
        points = tess.points.copy()
        # First calculate the tesselation
        #assert (IBZ.find_simplex(points) >= 0).all()

        with timer('Calculate nndist'):
            simplexpoints_spv = points[tess.simplices]
            s_spv = simplexpoints_spv
            nndist_s = np.array([md(s_spv[:, 0], s_spv[:, 1]),
                                 md(s_spv[:, 0], s_spv[:, 2]),
                                 md(s_spv[:, 0], s_spv[:, 3]),
                                 md(s_spv[:, 1], s_spv[:, 2]),
                                 md(s_spv[:, 1], s_spv[:, 3]),
                                 md(s_spv[:, 2], s_spv[:, 3])]).max(0)
            simplexpoints_spv = simplexpoints_spv[:, 1:] - \
                                simplexpoints_spv[:, 0][:, np.newaxis]
            volumes_s = np.abs(np.linalg.det(simplexpoints_spv))
            nndist_s[volumes_s < 1e-10] = 0.0

        if np.max(nndist_s) < nn_dist:
            break

        index = np.argmax(nndist_s)

        p_pv = points[tess.simplices[index]]
        norm_p = vectornorm(p_pv)
        center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                          norm_p[1:]**2 - norm_p[0]**2) / 2.

        newpoints = center_v[np.newaxis]
        if IBZ.find_simplex(center_v) < 0:
            newpoints = project_onto_3dhull(IBZhull, newpoints)
            cond = True
        else: 
            cond = True

        if cond:
            # Try to check whether the voronoi
            # decomposition is valid
            with timer('voronoi2'):
                vor.add_points(newpoints)
            vid = np.array(vor.regions[vor.point_region[-1]])
            vid = vid[vid != -1]
            vertices2_v = vor.vertices[vid]
            vertices2_v = vertices2_v[(np.abs(vertices2_v) < 1e10).all(-1)]

            notinibz = IBZ.find_simplex(vertices2_v) < 0
            if notinibz.any():
                vertices2_v = vertices2_v[notinibz]
                newpoints = refine_edge_or_face(IBZhull, vertices2_v)
                newpoints = [newpoints[0]]
                vor = Voronoi(points.copy(), incremental=True,
                              qhull_options='Qc Qx QJ Q11')
                vor.add_points(newpoints)

        pb.update(nn_dist / nndist_s[index])
        with timer('add points'):
            tess.add_points(newpoints)
    assert len(tess.coplanar) == 0

    tess.close()

    pb.finish()
    timer.write()
    tess.close()
    print('npoints', tess.npoints)
    points = unfold_points(tess.points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def tesselate_brillouin_eigs_del(calc, density=3.5):
    """Refine kpoint grid of previously calculated ground state."""
    if isinstance(calc, str):
        calcstr = calc
        calc = GPAW(calc, txt=None)

    ef = calc.get_fermi_level()
    shift = 0
    cell_cv = calc.wfs.gd.cell_cv
    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = cell_cv / (2 * np.pi)
    sym = calc.wfs.kd.symmetry

    # Tesselate lattice IBZ
    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)
    U_svv = np.dot(B_cv.T, np.dot(U_scc, A_cv)).transpose((1, 0, 2))
        
    bzk_kc, ibzk_kc = get_BZ(calc)
    ibzk_kv = np.dot(ibzk_kc, B_cv)

    # This could probably be optimized
    density = density / Bohr
    nn_dist = 1. / density
    IBZ = Delaunay(ibzk_kv)
    IBZhull = ConvexHull(ibzk_kv)
    points = IBZ.points.copy()

    # The Voronoi diagram connects the
    # circumcenters of the Delaunay diagram 
    timer = Timer()
    pb = ProgressBar()

    from scipy.spatial import minkowski_distance as md

    while True:
        # First calculate the tesselation
        assert (IBZ.find_simplex(points) >= 0).all()
        with timer('voronoi1'):
            tess = Delaunay(points)
            
        with timer('Calculate nndist'):
            simplexpoints_spv = points[tess.simplices]
            s_spv = simplexpoints_spv
            nndist_s = np.array([md(s_spv[:, 0], s_spv[:, 1]),
                                 md(s_spv[:, 0], s_spv[:, 2]),
                                 md(s_spv[:, 0], s_spv[:, 3]),
                                 md(s_spv[:, 1], s_spv[:, 2]),
                                 md(s_spv[:, 1], s_spv[:, 3]),
                                 md(s_spv[:, 2], s_spv[:, 3])]).max(0)
            simplexpoints_spv = simplexpoints_spv[:, 1:] - \
                                simplexpoints_spv[:, 0][:, np.newaxis]
            volumes_s = np.abs(np.linalg.det(simplexpoints_spv))
            nndist_s[volumes_s < 1e-10] = 0.0

        if np.max(nndist_s) < nn_dist:
            break

        index = np.argmax(nndist_s)
        
        p_pv = tess.points[tess.simplices[index]]
        norm_p = vectornorm(p_pv)
        center_v = np.dot(np.linalg.inv(p_pv[1:] - p_pv[0]),
                          norm_p[1:]**2 - norm_p[0]**2) / 2.

        newpoints = center_v[np.newaxis]
        if IBZ.find_simplex(center_v) < 0:
            newpoints = project_onto_3dhull(IBZhull, newpoints)

        # Try to check whether the voronoi
        # decomposition is valid
        ansatzpoints = np.concatenate([points, newpoints])
        with timer('voronoi2'):
            vor2 = Voronoi(ansatzpoints)
        vid = np.array(vor2.regions[vor2.point_region[-1]])
        vid = vid[vid != -1]
        vertices2_v = vor2.vertices[vid]
        vertices2_v = vertices2_v[(np.abs(vertices2_v) < 1e10).all(-1)]

        notinibz = IBZ.find_simplex(vertices2_v) < 0
        if notinibz.any():
            vertices2_v = vertices2_v[notinibz]
            newpoints = refine_edge_or_face(IBZhull, vertices2_v)
            newpoints = [newpoints[0]]

        newpoints = unique_rows(newpoints)
        # Do simple LCAO calculations for the
        # eigenvalues
        tmpcalc = GPAW(calcstr, mode='lcao',
                       basis='dzp', fixdensity=True,
                       kpts=newpoints, symmetry='off',
                       txt=None, nbands=-1)
        tmpcalc.get_potential_energy()

        mask = []
        for k in range(len(newpoints)):
            eigs_n = (tmpcalc.get_eigenvalues(kpt=k)
                      + shift - ef)
            neareigs_kn = np.array([eigs_kn[neark[k][0]],
                                    eigs_kn[neark[k][1]]])
            if ((eigs_n - np.sum(neareigs_kn, axis=0) / 2)
                < 0.1).all():
                continue

                mask.append(k)
            eigs_kn.append(eigs_n)
            newpoints = newpoints[mask]

        points = np.concatenate([points, newpoints])
        assert len(points) == len(unique_rows(points))
        pb.update(nn_dist / nndist_s[index])
    tess = Delaunay(points)
    assert len(tess.coplanar) == 0

    pb.finish()
    timer.write()
    points = unfold_points(points, U_svv)
    bzk_kc = np.dot(points, A_cv.T)
    bzk_kc = unique_rows(bzk_kc, tol=1e-8, mod=1)
    return bzk_kc


def refine_edge_or_face(hull, vertices):
    projectedvertices = project_onto_3dhull(hull,
                                            vertices)

    tmpvertices = project_onto_3dhull(hull,
                                      projectedvertices,
                                      only_onto_edges=True)
    mask = vectornorm(tmpvertices - projectedvertices) < 1e-8
    
    if mask.any():
        newpoints = projectedvertices[mask]
    else:
        newpoints = projectedvertices

    return newpoints


def project_onto_3dhull(hull, points_pv, return_indices=False,
                        project_onto_everything=False, only_onto_edges=False,
                        only_onto_faces=False):
    "Projects a set of points onto a convex hull."

    tess = Delaunay(hull.points)

    normal_fv = hull.equations[:, :-1]
    offset_f = hull.equations[:, -1]

    tmpfaces_fv = np.zeros((len(normal_fv), 4), float)
    tmpfaces_fv[:, :3] = np.array(normal_fv)
    tmpfaces_fv[:, 3] = np.array(offset_f)
    tmpfaces_fv = unique_rows(tmpfaces_fv)
    normal_fv = tmpfaces_fv[:, :3]
    offset_f = tmpfaces_fv[:, 3]

    lines_lv = []
    offset_lv  = []

    for f, (normal1_v, offset1) in enumerate(zip(normal_fv[:-1],
                                                 offset_f[:-1])):
        for normal2_v, offset2 in zip(normal_fv[f + 1:], offset_f[f + 1:]):
            line_v = np.cross(normal1_v, normal2_v)
            if (np.abs(line_v) < 1e-8).all():
                continue
            ind = np.argmax(np.abs(line_v))
            mask = np.ones(3, bool)
            mask[ind] = False

            M = np.array([normal1_v, normal2_v])[:, mask]
            offset_v = np.zeros(3, float)
            offset_v[mask] = - np.dot(np.linalg.inv(M), np.array([offset1, offset2]))

            offset_lv.append(offset_v)
            lines_lv.append(line_v)

    lines_lv = np.array(lines_lv)
    lines_lv /= vectornorm(lines_lv)[:, np.newaxis]

    offset_lv = np.array(offset_lv)

    hullpoints_pv = hull.points

    # First project onto planes
    dist_fp = - (np.dot(normal_fv, points_pv.T) + offset_f[:, np.newaxis])

    projectedpoints_fpv = (points_pv[np.newaxis]
                           + dist_fp[:, :, np.newaxis]
                           * normal_fv[:, np.newaxis])

    # Then onto lines
    dist_lp = np.dot(lines_lv, points_pv.T) - (lines_lv * offset_lv).sum(-1)[:, np.newaxis]

    projectedpoints_lpv = (dist_lp[:, :, np.newaxis] * lines_lv[:, np.newaxis] +
                           offset_lv[:, np.newaxis])

    if only_onto_edges and only_onto_faces:
        projectedpoints_xpv = np.concatenate([projectedpoints_fpv,
                                              projectedpoints_lpv])
    elif only_onto_edges:
        projectedpoints_xpv = projectedpoints_lpv
    elif only_onto_faces:
        projectedpoints_xpv = projectedpoints_fpv
    else:
        projectedpoints_xpv = np.concatenate([projectedpoints_fpv,
                                              projectedpoints_lpv,
                                              np.tile(hullpoints_pv[:, np.newaxis],
                                                      (1, len(points_pv), 1))])

    if project_onto_everything:
        points = projectedpoints_xpv.reshape(-1, 3)
        return points

    notinibzprojected_xp = (tess.find_simplex(projectedpoints_xpv.reshape(-1, 3),
                                              tol=1e-8)
                            < 0).reshape(projectedpoints_xpv.shape[:-1])

    projectedpoints_xpv[np.where(notinibzprojected_xp)] = np.inf
    dist_xp = vectornorm(projectedpoints_xpv - points_pv)
    indices_p = np.argmin(dist_xp, axis=0)
    projectedpoints_pv = projectedpoints_xpv[indices_p,
                                             np.arange(0, len(points_pv)), :]

    return projectedpoints_pv

def vectornorm(A):
    return (A**2).sum(-1)**0.5

def unique_rows(ain, tol=1e-10, mod=None, aglomerate=True):
    # Move to positive octant
    a = ain - ain.min(0)

    # First take modulus
    if mod is not None:
        a = np.mod(np.mod(a, mod), mod)
    
    # Round and take modulus again
    if aglomerate:
        aglomerate_points(a, tol)
    a = a.round(-np.log10(tol).astype(int))
    if mod is not None:
        a = np.mod(a, mod)

    # Now perform ordering
    order = np.lexsort(a.T)
    a = a[order]

    # Find unique rows
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(1)

    return ain[order][ui]


def get_smallest_Gvecs(cell_cv, n=5):
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2
    G_xv = np.dot(N_xc, B_cv)

    return G_xv, N_xc


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
    G_xv, N_xc = get_smallest_Gvecs(cell_cv, n=n)
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
    ibzk_kc = expand_ibz_new(lU_scc, cU_scc, ibzk_kc)

    # Fold out to full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               cU_scc.transpose(0, 2, 1))))

    return bzk_kc, ibzk_kc


def get_reduced_BZ2(cell_cv, cU_scc, time_reversal):
    # Find Lattice IBZ
    ibzk_kc = get_IBZ_vertices(cell_cv,
                               U_scc=cU_scc,
                               time_reversal=time_reversal)

    # Fold out to full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               cU_scc.transpose(0, 2, 1))))

    return bzk_kc, ibzk_kc


def expand_ibz(lU_scc, cU_scc, ibzk_kc):
    # Find right cosets
    cosets = []
    Utmp_scc = lU_scc.copy()
    while len(Utmp_scc):
        U1_cc = Utmp_scc[0].copy()
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
        if len(nibzk_ikc):
            tmp_kc = unique_rows(np.concatenate(nibzk_ikc))
        else:
            tmp_kc = ibzk_kc
        ibzk_ksc = np.dot(tmp_kc, np.array(U_scc).transpose((0, 2, 1)))
        dist_ks = ((ibzk_ksc -
                    tmp_kc[:, np.newaxis])**2).sum(-1)**0.5
        dist_s = dist_ks.sum(0)
        s = np.argmin(dist_s)
        U_cc = U_scc[s]
        nibzk_ikc.append(np.dot(ibzk_kc, U_cc.T))

    ibzk_kc = unique_rows(np.concatenate(nibzk_ikc))

    return ibzk_kc


def expand_ibz_new(lU_scc, cU_scc, ibzk_kc):
    # Find right cosets
    cosets = []
    Utmp_scc = lU_scc.copy()
    while len(Utmp_scc):
        U1_cc = Utmp_scc[0].copy()
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

    from itertools import product

    volume = np.inf
    nibzk_kc = ibzk_kc
    U0_cc = cosets[0][0]  # Origin

    from gpaw.mpi import world

    combs = list(product(*cosets[1:]))[world.rank::world.size]
    for U_scc in combs:
        if not len(U_scc):
            continue
        U_scc = np.concatenate([np.array(U_scc), [U0_cc]])
        tmpk_kc = unfold_points(ibzk_kc, U_scc)
        volumenew = convex_hull_volume(tmpk_kc)

        if volumenew < volume:
           nibzk_kc = tmpk_kc
           volume = volumenew

    ibzk_kc = unique_rows(nibzk_kc)
    volume = np.array((volume,))

    volumes = np.zeros(world.size, float)
    world.all_gather(volume, volumes)

    minrank = np.argmin(volumes)
    minshape = np.array(ibzk_kc.shape)
    world.broadcast(minshape, minrank)

    if world.rank != minrank:
        ibzk_kc = np.zeros(minshape, float)
    world.broadcast(ibzk_kc, minrank)

    return ibzk_kc


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6


def convex_hull_volume(pts):
    dt = Delaunay(pts)
    tets = dt.points[dt.simplices]
    vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], 
                                    tets[:, 2], tets[:, 3]))
    return vol


def convex_hull_volume_bis(pts):
    ch = ConvexHull(pts)

    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))


def rearrange_band_path(calc):
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)
    nbands = calc.get_number_of_bands()
    kpts = calc.get_ibz_k_points()
    nkpts = len(kpts)
    
    B_cv = calc.wfs.gd.icell_cv * 2 * np.pi

    kpts = np.dot(kpts, B_cv)
    
    eigs_nk = np.empty((nbands, nkpts), float)
    
    for k in range(nkpts):
        eigs_nk[:, k] = calc.get_eigenvalues(kpt=k)
    
    for k1, k2, k3 in zip(range(nkpts - 2), range(1, nkpts - 1),
                          range(2, nkpts)):
        e1_n = eigs_nk[:, k1]
        e2_n = eigs_nk[:, k2]
        e3_n = eigs_nk[:, k3]
        k1_c = kpts[k1]
        k2_c = kpts[k2]
        k3_c = kpts[k3]
        dedk1_n = (e2_n - e1_n) / (((k2_c - k1_c)**2).sum())**0.5
        dedk2_nn = (e3_n[np.newaxis] - e2_n[:, np.newaxis]) / (((k3_c - k2_c)**2).sum())**0.5
        diff_nn = np.abs(dedk2_nn - dedk1_n[:, np.newaxis])
        order_N = np.argsort(diff_nn.ravel())
    
        n1_n, n2_n = np.unravel_index(order_N, (nbands, nbands))
        order_n = np.array([-1] * nbands, int)
    
        for n1, n2 in zip(n1_n, n2_n):
            if n2 not in order_n and order_n[n1] == -1:
                order_n[n1] = n2

        eigs_nk[:, k3] = eigs_nk[order_n, k3]

    for ik in range(nkpts):
        calc.wfs.kpt_u[ik].eps_n[:] = eigs_nk[:, ik]

    return calc


def rearrange_bands(calc, nd=3, do_wfs=False, seed=0):
    """Sort bands.

    calc: str or GPAW instance
    do_wfs: bool
        Switch for sorting wavefunctions as well
    """

    if isinstance(calc, str):
        calc = GPAW(calc, fixdensity=True,
                    communicator=mpi.serial_comm, txt=None)

    wfs = calc.wfs
    kd = wfs.kd
    gd = wfs.gd
    sym = kd.symmetry

    B_cv = gd.icell_cv * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T

    U_scc = get_symmetry_operations(sym.op_scc,
                                    sym.time_reversal)

    # Determine the irreducible BZ
    bzk_kc, ibzk_kc = get_reduced_BZ(gd.cell_cv,
                                     U_scc,
                                     False)

    # Sample a larger region of reciprocal space
    n = 3  # This is sufficient for most cases
    N_xc = (np.indices((n, n, n)).reshape((3, n**3)).T
            - n // 2)  # Reciprocal lattice vector
    IBZ = Delaunay(ibzk_kc)
    k_kc = []
    for N_c in N_xc:
        tmpk_kc = kd.bzk_kc + N_c
        tmpk_kc = tmpk_kc[IBZ.find_simplex(tmpk_kc) >= 0]
        if not len(k_kc) and len(tmpk_kc):
            k_kc = unique_rows(tmpk_kc)
        elif len(k_kc):
            k_kc = unique_rows(np.append(tmpk_kc, k_kc, axis=0))

    k_kv = np.dot(k_kc, B_cv)
    ibzk_kv = np.dot(ibzk_kc, B_cv)
    IBZhull = ConvexHull(ibzk_kv)

    nkpts = len(k_kv)
    assert nkpts == len(kd.ibzk_kc)
    tree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1), 1))
    tess = Delaunay(k_kv)
    nbands = calc.get_number_of_bands()
    eigs_nk = np.zeros((nbands, nkpts), float)

    def find_iu(k_v):
        k_c = np.dot(k_v, A_cv.T)
        K = tree.query(np.mod(np.mod(k_c, 1), 1))[1]
        iu = kd.bz2ibz_k[K]
        return iu

    for ik, k_v in enumerate(k_kv):
        iu = find_iu(k_v)
        try:
            eigs_nk[:, ik] = calc.wfs.kpt_u[iu].eps_n
        except:
            print(iu)
            raise

    neighbors = tess.neighbors

    pts_k = [[] for p in xrange(tess.npoints)]
    for s, K_k in enumerate(tess.simplices):
        for K in K_k:
            pts_k[K].append(s)

    # Change to numpy arrays:
    for k in xrange(tess.npoints):
        pts_k[k] = np.array(pts_k[k], int)

    neighborhood = [[] for s in range(tess.nsimplex)]
    for s, simplex in enumerate(tess.simplices):
        for point in simplex:
            neighborhood[s].extend(pts_k[point].tolist())

    from scipy.stats import itemfreq
    for s in range(tess.nsimplex):
        neighborstmp = np.array(neighborhood[s])
        freqs = itemfreq(neighborstmp)
        nbhs = set(freqs[:, 0][freqs[:, 1] >= 1].astype(int).tolist())
        nbhs.discard(s)
        neighborhood[s] = nbhs

    seed = tess.find_simplex(ibzk_kv.sum(0) / len(ibzk_kv)).tolist()

    assert seed != -1

    # Data-structures for performing the ordering
    grad_snv = np.zeros((tess.nsimplex, nbands, nd), float)  # Store gradients
    priority_s = np.zeros(tess.nsimplex, int)  # Higher int => larger priority
    frontiersimplices = set([seed])  # Choose somewhere arbitrary to start
    interiorsimplices = set()  # Simplices that have alredy been treated
    surfacesimplices = set()
    badinteriorsimplices = set()
    sortedpoints = set()  # KPoints that have been sorted

    timer = Timer()

    pb = ProgressBar()

    np.set_printoptions(precision=4, suppress=True)

    i = 0
    while frontiersimplices:
        pb.update(i / tess.nsimplex)

        # Take the simplex with the highest priority
        if frontiersimplices - surfacesimplices:
            tmpfront = list(frontiersimplices - surfacesimplices)
        else:
            tmpfront = list(frontiersimplices)

        simplex = tmpfront[priority_s[tmpfront].argmax()]

        points = tess.simplices[simplex]
        k_kv = tess.points[points]
        kprojected_kv = project_onto_3dhull(IBZhull, k_kv)

        onhull = (vectornorm(kprojected_kv - k_kv) < 1e-10).sum() > 0

        if onhull and frontiersimplices - surfacesimplices:
            surfacesimplices.add(simplex)
            continue

        i += 1
        priority_s[tmpfront] += 1

        frontiersimplices.discard(simplex)
        interiorsimplices.add(simplex)

        # If this is a bad simplex
        # just continue to next simplex
        A_kv = np.append(tess.points[points],
                         np.ones(4)[:, np.newaxis], axis=1)

        D_kv = np.append((A_kv[:, :-1]**2).sum(1)[:, np.newaxis],
                         A_kv, axis=1)
        a = np.linalg.det(D_kv[:, np.arange(5) != 0])

        Dx = np.linalg.det(D_kv[:, np.arange(5) != 1])
        Dy = np.linalg.det(D_kv[:, np.arange(5) != 2])
        Dz = np.linalg.det(D_kv[:, np.arange(5) != 3])
        c = np.linalg.det(D_kv[:, np.arange(5) != 4])
        if np.abs(a) < 1e-10:
            badinteriorsimplices.add(simplex)
            continue

        radius = (Dx**2 + Dy**2 + Dz**2 - 4 * a * c)**0.5 / (2 * np.abs(a))
        ratio = radius / np.abs(a)**(1. / 3)

        #if ratio > 2:
        #    badinteriorsimplices.add(simplex)
        #    continue

        # Find the neighbors
        neighbors_s = (set(neighbors[simplex])
                       - set([-1]) - set([simplex]))
        frontiersimplices.update(neighbors_s - interiorsimplices)

        # Update priority
        priority_s[list(neighbors_s)] += 1

        es_nk = eigs_nk[:, points]
        grad_nv = np.dot(es_nk[:, 1:] - es_nk[:, [0]],
                         np.linalg.inv(k_kv[1:] - k_kv[0]).T)
        grad_snv[simplex] = grad_nv
        sortedneighbours_s = interiorsimplices & (neighborhood[simplex] -
                                                  badinteriorsimplices)


        if len(sortedneighbours_s):
            try:
                unsortedpoint = (set(points) - sortedpoints)
                assert len(unsortedpoint) < 2, print(unsortedpoint)
                point = unsortedpoint.pop()
            except KeyError:
                continue

            k_v = tess.points[point]
            iu = find_iu(k_v)

            sortedpoints.add(point)
            invk_kv = np.linalg.inv(k_kv[points != point] -
                                    k_kv[points == point]).T
            grad_nnv = np.dot(es_nk[:, points != point][:, np.newaxis] -
                              es_nk[:, points == point][np.newaxis],
                              invk_kv)
            ngrad_snv = grad_snv[list(sortedneighbours_s)]

            diff_nn = vectornorm(ngrad_snv[:, :, np.newaxis, :] -
                                 grad_nnv[np.newaxis]).sum(0)

            order_N = np.argsort(diff_nn.ravel())
            n1_n, n2_n = np.unravel_index(order_N, (nbands, nbands))

            order_n = np.array([-1] * nbands, int)
            count = 0
            for n1, n2 in zip(n1_n, n2_n):
                if order_n[n1] == -1 and (order_n - n2).all():
                    order_n[n1] = n2
                    count += 1
                    if count == nbands:
                        break

            assert (order_n != -1).all(), print(order_n)
            assert len(np.unique(order_n)) == nbands

            eigs_nk[:, point] = eigs_nk[:, point].copy()[order_n]
            es_nk = eigs_nk[:, points]
            calc.wfs.kpt_u[iu].eps_n[:] = calc.wfs.kpt_u[iu].eps_n.copy()[order_n]
            calc.wfs.kpt_u[iu].f_n[:] = calc.wfs.kpt_u[iu].f_n.copy()[order_n]

            if do_wfs:
                P_ani = calc.wfs.kpt_u[iu].P_ani.copy()
                for a in P_ani:
                    P_ni = P_ani[a].copy()
                    P_ani[a] = P_ni[order_n]
    
                calc.wfs.kpt_u[iu].P_ani = P_ani
                if calc.wfs.kpt_u[iu].psit_nG:
                    calc.wfs.kpt_u[iu].psit_nG = calc.wfs.kpt_u[iu].psit_nG[:][order_n, :].copy()
        else:
            sortedpoints.update(points)

        grad_nv = np.dot(es_nk[:, 1:] - es_nk[:, [0]],
                         np.linalg.inv(k_kv[1:] - k_kv[[0]]).T)

        grad_snv[simplex] = grad_nv

    print(tess.coplanar)

    assert len(sortedpoints) == tess.npoints, print(len(sortedpoints), tess.npoints)

    pb.finish()

    timer.write()

    return calc

