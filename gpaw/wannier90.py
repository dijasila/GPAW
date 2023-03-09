import numpy as np
from gpaw.utilities.blas import gemmdot
from gpaw.berryphase import get_overlap
from gpaw.response.pair import KPoint
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.context import ResponseContext
# XXX. Wannier90 class is useless at the moment.
# should put functions onto class instead.

class Wannier90:
    def __init__(self, calc, seed=None, bands=None, orbitals_ai=None,
                 spin=0):

        if seed is None:
            seed = calc.atoms.get_chemical_formula()
        self.seed = seed

        if bands is None:
            bands = range(calc.get_number_of_bands())
        self.bands = bands

        Na = len(calc.atoms)
        if orbitals_ai is None:
            orbitals_ai = []
            for ia in range(Na):
                ni = 0
                setup = calc.wfs.setups[ia]
                for l, n in zip(setup.l_j, setup.n_j):
                    if not n == -1:
                        ni += 2 * l + 1
                orbitals_ai.append(range(ni))

        self.calc = calc
        self.bands = bands
        self.Nn = len(bands)
        self.Na = Na
        self.orbitals_ai = orbitals_ai
        self.Nw = np.sum([len(orbitals_ai[ai]) for ai in range(Na)])
        self.kpts_kc = calc.get_ibz_k_points() # XXX If used should be updated
        self.Nk = len(self.kpts_kc)            # work with symmetry
        self.spin = spin


def write_input(calc,
                seed=None,
                bands=None,
                orbitals_ai=None,
                mp=None,
                plot=False,
                num_iter=100,
                write_xyz=False,
                write_rmn=False,
                translate_home_cell=False,
                dis_num_iter=200,
                dis_froz_max=0.1,
                dis_mix_ratio=0.5,
                search_shells=None,
                spinors=False,
                write_u_matrices=False):
    if spinors:
        # XXX If class function this assertion could be moved to class
        assert calc.wfs.kd.nibzkpts == calc.wfs.kd.nbzkpts
    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    if bands is None:
        bands = range(calc.get_number_of_bands())

    Na = len(calc.atoms)
    if orbitals_ai is None:
        orbitals_ai = []
        for ia in range(Na):
            ni = 0
            setup = calc.wfs.setups[ia]
            for l, n in zip(setup.l_j, setup.n_j):
                if not n == -1:
                    ni += 2 * l + 1
            orbitals_ai.append(range(ni))
    assert len(orbitals_ai) == Na

    Nw = np.sum([len(orbitals_ai[ai]) for ai in range(Na)])
    if spinors:
        Nw *= 2
        new_bands = []
        for n in bands:
            new_bands.append(2 * n)
            new_bands.append(2 * n + 1)
        bands = new_bands

    f = open(seed + '.win', 'w')

    pos_ac = calc.spos_ac
    # pos_av = calc.atoms.get_positions()
    # cell_cv = calc.atoms.get_cell()
    # pos_ac = np.dot(pos_av, np.linalg.inv(cell_cv))

    print('begin projections', file=f)
    for ia, orbitals_i in enumerate(orbitals_ai):
        setup = calc.wfs.setups[ia]
        l_i = []
        n_i = []
        for n, l in zip(setup.n_j, setup.l_j):
            if not n == -1:
                l_i += (2 * l + 1) * [l]
                n_i += (2 * l + 1) * [n]
        r_c = pos_ac[ia]
        for orb in orbitals_i:
            l = l_i[orb]
            n = n_i[orb]
            print('f=%1.2f, %1.2f, %1.2f : s ' % (r_c[0], r_c[1], r_c[2]),
                  end='', file=f)
            print('# n = %s, l = %s' % (n, l), file=f)

    print('end projections', file=f)
    print(file=f)

    if spinors:
        print('spinors = True', file=f)
    else:
        print('spinors = False', file=f)
    if write_u_matrices:
        print('write_u_matrices = True', file=f)
    print('write_hr = True', file=f)
    if write_xyz:
        print('write_xyz = True', file=f)
    if write_rmn:
        print('write_tb = True', file=f)
        print('write_rmn = True', file=f)
    if translate_home_cell:
        print('translate_home_cell = True', file=f)
    print(file=f)
    print('num_bands       = %d' % len(bands), file=f)

    if search_shells is not None:
        print("search_shells = {0}".format(search_shells), file=f)

    maxn = max(bands)
    if maxn + 1 != len(bands):
        diffn = maxn - len(bands)
        print('exclude_bands : ', end='', file=f)
        counter = 0
        for n in range(maxn):
            if n not in bands:
                counter += 1
                if counter != diffn + 1:
                    print('%d,' % (n + 1), sep='', end='', file=f)
                else:
                    print('%d' % (n + 1), file=f)
    print(file=f)

    print('guiding_centres = True', file=f)
    print('num_wann        = %d' % Nw, file=f)
    print('num_iter        = %d' % num_iter, file=f)
    print(file=f)

    if len(bands) > Nw:
        ef = calc.get_fermi_level()
        print('fermi_energy  = %2.3f' % ef, file=f)
        if abs(dis_froz_max) > 0.0:
            print('dis_froz_max  = %2.3f' % (ef + dis_froz_max), file=f)
        print('dis_num_iter  = %d' % dis_num_iter, file=f)
        print('dis_mix_ratio = %1.1f' % dis_mix_ratio, file=f)
    print(file=f)

    print('begin unit_cell_cart', file=f)
    for cell_c in calc.atoms.cell:
        print('%14.10f %14.10f %14.10f' % (cell_c[0], cell_c[1], cell_c[2]),
              file=f)
    print('end unit_cell_cart', file=f)
    print(file=f)

    print('begin atoms_frac', file=f)
    for atom, pos_c in zip(calc.atoms, pos_ac):
        print(atom.symbol, end='', file=f)
        print('%14.10f %14.10f %14.10f' % (pos_c[0], pos_c[1], pos_c[2]),
              file=f)
    print('end atoms_frac', file=f)
    print(file=f)

    if plot:
        print('wannier_plot   = True', file=f)
        print('wvfn_formatted = True', file=f)
        print(file=f)

    if mp is not None:
        N_c = mp
    else:
        N_c = calc.wfs.kd.N_c
    print('mp_grid =', N_c[0], N_c[1], N_c[2], file=f)
    print(file=f)
    print('begin kpoints', file=f)

    for kpt in calc.get_bz_k_points():
        print('%14.10f %14.10f %14.10f' % (kpt[0], kpt[1], kpt[2]), file=f)
    print('end kpoints', file=f)

    f.close()

def get_P_ani(calc, ik, spin = 0, spinors = False, soc = None):
    """Returns P_ani
    calc: GPAW calculator
    ik: BZ k-point index
    spin: spin index
    kpt: gpaw.response.pair KPoint object
    spinors: logical
    soc: SOC object (if spinors)
    """
    if spinors:
        assert soc is not None
        P_ani = soc[ik].P_amj
    else:
        n2 = calc.get_number_of_bands()
        kpt = KPoint.get_k_point(gs, context.timer,
                                 spin, ik,
                                 0, n2)
        P_ani = kpt.P_ani #calc.wfs.kpt_qs[ik][spin].P_ani
    return P_ani

def get_gs_and_context(calc, seed, spinors):
    if spinors:
        return None, None
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext(txt=seed+'.txt')
    return gs, context

def write_projections(calc, seed=None, spin=0, orbitals_ai=None, soc=None):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()
    
    bands = get_bands(seed)
    Nn = len(bands)

    spinors = False

    win_file = open(seed + '.win')
    for line in win_file.readlines():
        l_e = line.split()
        if len(l_e) > 0:
            if l_e[0] == 'spinors':
                spinors = l_e[2]
                if spinors in ['T', 'true', '1', 'True']:
                    spinors = True
                else:
                    spinors = False
            if l_e[0] == 'num_wann':
                Nw = int(l_e[2])
            if l_e[0] == 'mp_grid':
                Nk = int(l_e[2]) * int(l_e[3]) * int(l_e[4])
                assert Nk == len(calc.get_bz_k_points())
                
    # get stuff needed for kpt descriptor
    gs, context = get_gs_and_context(calc, seed, spinors)
    Na = len(calc.atoms)
    if orbitals_ai is None:
        orbitals_ai = []
        for ia in range(Na):
            ni = 0
            setup = calc.wfs.setups[ia]
            for l, n in zip(setup.l_j, setup.n_j):
                if not n == -1:
                    ni += 2 * l + 1
            orbitals_ai.append(range(ni))
    assert len(orbitals_ai) == Na

    if spinors:
        new_orbitals_ai = []
        for orbitals_i in orbitals_ai:
            new_orbitals_i = []
            for i in orbitals_i:
                new_orbitals_i.append(2 * i)
                new_orbitals_i.append(2 * i + 1)
            new_orbitals_ai.append(new_orbitals_i)
        orbitals_ai = new_orbitals_ai

    Ni = 0
    for orbitals_i in orbitals_ai:
        Ni += len(orbitals_i)
    assert Nw == Ni

    f = open(seed + '.amn', 'w')

    print('Kohn-Sham input generated from GPAW calculation', file=f)
    print('%10d %6d %6d' % (Nn, Nk, Nw), file=f)

    P_kni = np.zeros((Nk, Nn, Nw), complex)
    for ik in range(Nk):
        P_ani = get_P_ani(calc, ik, spin, spinors, soc)
        for i in range(Nw):
            icount = 0
            for ai in range(Na):
                ni = len(orbitals_ai[ai])
                P_ni = P_ani[ai][bands]
                P_ni = P_ni[:, orbitals_ai[ai]]
                P_kni[ik, :, icount:ni + icount] = P_ni.conj()
                icount += ni

    for ik in range(Nk):
        for i in range(Nw):
            for n in range(Nn):
                P = P_kni[ik, n, i]
                data = (n + 1, i + 1, ik + 1, P.real, P.imag)
                print('%4d %4d %4d %18.12f %20.12f' % data, file=f)

    f.close()


def write_eigenvalues(calc, seed=None, spin=0, soc=None):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    bands = get_bands(seed)
    if soc is None:
        # get stuff needed for kpt descriptor
        gs, context = get_gs_and_context(calc, seed, spinors=False)

    
    f = open(seed + '.eig', 'w')

    for ik in range(len(calc.get_bz_k_points())):
        if soc is None:
            n2 = calc.get_number_of_bands()
            kpt = KPoint.get_k_point(gs, context.timer,
                                     spin, ik,
                                     0, n2)

            e_n = kpt.eps_n #calc.get_eigenvalues(kpt=ik, spin=spin)
        else:
            e_n = soc[ik].eig_m
        for i, n in enumerate(bands):
            data = (i + 1, ik + 1, e_n[n])
            print('%5d %5d %14.6f' % data, file=f)

    f.close()

def get_wf(bz_index, bands, spinors, soc=None, calc=None,
           spin = None, gs=None, context=None):
    if spinors:
        # For spinors, G denotes spin and grid: G = (s, gx, gy, gz)
        return soc[bz_index].wavefunctions(
            calc, periodic=True)[bands]
    # For non-spinors, G denotes grid: G = (gx, gy, gz)
    #if kpt is None:
    n1 = bands[0]
    n2 = bands[-1]+1
    kpt = KPoint.get_k_point(gs, context.timer,
                             spin, bz_index,
                             n1, n2)
    return kpt.get_shifted_ut_nR()#kpt.ut_nR #kpt.get_u_nG(bands = bands)


def write_overlaps(calc, seed=None, spin=0, soc=None, less_memory=False):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    if soc is None:
        spinors = False
    else:
        spinors = True
    gs, context = get_gs_and_context(calc, seed, spinors)
    bands = get_bands(seed)
    Nn = len(bands)
    kpts_kc = calc.get_bz_k_points()
    Nk = len(kpts_kc)

    nnkp = open(seed + '.nnkp', 'r')
    lines = nnkp.readlines()
    for il, line in enumerate(lines):
        if len(line.split()) > 1:
            if line.split()[0] == 'begin' and line.split()[1] == 'nnkpts':
                Nb = eval(lines[il + 1].split()[0])
                i0 = il + 2
                break

    f = open(seed + '.mmn', 'w')

    print('Kohn-Sham input generated from GPAW calculation', file=f)
    print('%10d %6d %6d' % (Nn, Nk, Nb), file=f)

    icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T
    r_g = calc.wfs.gd.get_grid_point_coordinates()
    Ng = np.prod(np.shape(r_g)[1:]) * (spinors + 1)

    dO_aii = []
    for ia in calc.wfs.kpt_u[0].P_ani.keys():
        dO_ii = calc.wfs.setups[ia].dO_ii
        if spinors:
            # Spinor projections require doubling of the (identical) orbitals
            dO_jj = np.zeros((2 * len(dO_ii), 2 * len(dO_ii)), complex)
            dO_jj[::2, ::2] = dO_ii
            dO_jj[1::2, 1::2] = dO_ii
            dO_aii.append(dO_jj)
        else:
            dO_aii.append(dO_ii)

    wfs = calc.wfs
    
    def wavefunctions(bz_index):
        return get_wf(bz_index, bands, spinors, soc, calc,
                      spin, gs, context)
    
    #if not less_memory:
    #    u_knG = []
    #    for ik in range(Nk):
    #        u_nG = wavefunctions(ik)
    #        u_knG.append(u_nG)

    P_kani = []
    if not less_memory:
        u_knG = []

    for ik in range(Nk):
        P_ani = get_P_ani(calc, ik, spin, spinors, soc)
        P_kani.append(P_ani)
    
        if not less_memory:
            u_nG = wavefunctions(ik) #XXX Misleading name. It is actually u_nR!
            u_knG.append(u_nG)

    for ik1 in range(Nk):
        if less_memory:
            u1_nG = wavefunctions(ik1)
        else:
            u1_nG = u_knG[ik1]
        for ib in range(Nb):
            # b denotes nearest neighbor k-points
            line = lines[i0 + ik1 * Nb + ib].split()
            ik2 = int(line[1]) - 1
            if less_memory:
                u2_nG = wavefunctions(ik2)
            else:
                u2_nG = u_knG[ik2]
            #XXX Do I need shift_c here somewhere?
            #shift_c = np.dot(U_cc, ik_c) - k_c * sign
            #SO the symmetry transformed quantities are actually
            #at the k-point ik_c + shift_c != k_c
            G_c = np.array([int(line[i]) for i in range(2, 5)])
            bG_v = np.dot(G_c, icell_cv)
            u2_nG = u2_nG * np.exp(-1.0j * gemmdot(bG_v, r_g, beta=0.0))
            bG_c = kpts_kc[ik2] - kpts_kc[ik1] + G_c
            bG_v = np.dot(bG_c, icell_cv)  # Overwrite bG_v
            M_mm = get_overlap(calc,
                               bands,
                               np.reshape(u1_nG, (len(u1_nG), Ng)),
                               np.reshape(u2_nG, (len(u2_nG), Ng)),
                               P_kani[ik1],
                               P_kani[ik2],
                               dO_aii,
                               bG_v)
            indices = (ik1 + 1, ik2 + 1, G_c[0], G_c[1], G_c[2])
            print('%3d %3d %4d %3d %3d' % indices, file=f)
            for m1 in range(len(M_mm)):
                for m2 in range(len(M_mm)):
                    M = M_mm[m2, m1]
                    print('%20.12f %20.12f' % (M.real, M.imag), file=f)

    f.close()


def get_bands(seed):
    win_file = open(seed + '.win')
    exclude_bands = None
    for line in win_file.readlines():
        l_e = line.split()
        if len(l_e) > 0:
            if l_e[0] == 'num_bands':
                Nn = int(l_e[2])
            if l_e[0] == 'exclude_bands':
                exclude_bands = line.split()[2]
                exclude_bands = [int(n) - 1 for n in exclude_bands.split(',')]
    if exclude_bands is None:
        bands = range(Nn)
    else:
        bands = range(Nn + len(exclude_bands))
        bands = [n for n in bands if n not in exclude_bands]
    win_file.close()

    return bands


def write_wavefunctions(calc, soc=None, spin=0, seed=None):

    wfs = calc.wfs

    if soc is None:
        spinors = False
    else:
        spinors = True

    if seed is None:
        seed = calc.atoms.get_chemical_formula()
    gs, context = get_gs_and_context(calc, seed, spinors)
    bands = get_bands(seed)
    Nn = len(bands)
    Nk = len(calc.get_bz_k_points()) #XXX Also needs to be updated to work with symmetry
    Nk_ibz = len(calc.get_ibz_k_points())
    if spinors:
        assert Nk == Nk_ibz
        
    for ik in range(Nk):
        u_nG = get_wfs(bz_index, bands, spinors, soc, calc,
                       spin, gs, context)
        f = open('UNK%s.%d' % (str(ik + 1).zfill(5), spin + 1), 'w')
        grid_v = np.shape(u_nG)[1:]
        print(grid_v[0], grid_v[1], grid_v[2], ik + 1, Nn, file=f)
        for n in range(Nn):
            for iz in range(grid_v[2]):
                for iy in range(grid_v[1]):
                    for ix in range(grid_v[0]):
                        u = u_nG[n, ix, iy, iz]
                        print(u.real, u.imag, file=f)
        f.close()
