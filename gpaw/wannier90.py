from __future__ import print_function
import numpy as np
import datetime
from gpaw.utilities.blas import gemmdot
from ase.units import Bohr

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
        self.kpts_kc = calc.get_ibz_k_points()
        self.Nk = len(self.kpts_kc)
        self.spin = spin
        
def write_input(calc, 
                seed=None,
                bands=None,
                orbitals_ai=None,
                mp=None): 
    
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
    
    f = open(seed + '.win', 'w')

    pos_ac = calc.atoms.get_scaled_positions()

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

    Nw = np.sum([len(orbitals_ai[ai]) for ai in range(Na)])

    print('end projections', file=f)
    print(file=f)

    print('spinors = False', file=f)
    print('hr_plot = True', file=f)
    print(file=f)
    print('num_bands       = %d' % len(bands), file=f)

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
    print('num_iter        = 100', file=f)
    print(file=f)

    if len(bands) > Nw:
        ef = calc.get_fermi_level()
        print('dis_froz_max = %2.1f # (ef + 0.1)' % (ef + 0.1), file=f)
        print('dis_num_iter = 200', file=f)
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

def write_projections(calc, seed=None, spin=0, orbitals_ai=None):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    bands = get_bands(seed)
    Nn = len(bands)

    win_file = open(seed  + '.win')
    for line in win_file.readlines():
        l_e = line.split()
        if len(l_e) > 0:
            if l_e[0] == 'num_wann':
                Nw = int(l_e[2])
            if l_e[0] == 'mp_grid':
                Nk = int(l_e[2]) * int(l_e[3]) * int(l_e[4])
                assert Nk == len(calc.get_bz_k_points())

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
    Ni = 0
    for orbitals_i in orbitals_ai:
        Ni += len(orbitals_i)
    assert Nw == Ni

    f = open(seed + '.amn', 'w')

    print('Kohn-Sham input generated from GPAW calculation', file=f)
    print('%10d %6d %6d' % (Nn, Nk, Nw), file=f)

    P_kni = np.zeros((Nk, Nn, Nw), complex)
    for k in range(Nk):
        for i in range(Nw):
            icount = 0
            P_ani = calc.wfs.kpt_u[spin * Nk + k].P_ani
            for ai in range(Na):
                ni = len(orbitals_ai[ai])
                P_ni = P_ani[ai][bands]#, self.orbitals_ai[ai]]
                P_ni = P_ni[:, orbitals_ai[ai]]
                P_kni[k, :, icount:ni + icount] = P_ni.conj()
                icount += ni

    for k in range(Nk):
        for i in range(Nw):
            for n in range(Nn):
                P = P_kni[k, n, i]
                data = (n + 1, i + 1, k + 1, P.real, P.imag)
                print('%4d %4d %4d %18.12f %20.12f' % data, file=f)

    f.close()

def write_eigenvalues(calc, seed=None, spin=0):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    bands = get_bands(seed)

    f = open(seed + '.eig', 'w')

    for ik in range(len(calc.get_bz_k_points())):
        e_n = calc.get_eigenvalues(kpt=ik, spin=spin)
        for i, n in enumerate(bands):
            data = (i + 1, ik + 1, e_n[n])
            print('%5d %5d %14.6f' % data, file=f)

    f.close()

def write_overlaps(calc, seed=None, spin=0):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    bands = get_bands(seed)
    Nn = len(bands)
    kpts_kc = calc.get_bz_k_points()
    Nk = len(kpts_kc)

    nnkp = open(seed  + '.nnkp', 'r')
    lines = nnkp.readlines()
    neighbor_kb = []
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
    Ng = np.prod(np.shape(r_g)[1:])
    wfs = calc.wfs
    Nk = Nk
    for ik1 in range(Nk):
        u1_nG = np.array([wfs.get_wave_function_array(n, ik1, spin)
                          for n in bands])

        for ib in range(Nb):
            line = lines[i0 + ik1 * Nb + ib].split()
            ik2 = int(line[1]) - 1
            u2_nG = np.array([wfs.get_wave_function_array(n, ik2, spin)
                              for n in bands])

            G_c = np.array([int(line[i]) for i in range(2, 5)])
            bG_c = kpts_kc[ik2] - kpts_kc[ik1] + G_c
            bG_v = np.dot(bG_c, icell_cv)
            u2_nG = u2_nG * np.exp(-1.0j * gemmdot(bG_v, r_g, beta=0.0))
            isk1 = Nk * spin + ik1
            isk2 = Nk * spin + ik2
            M_mm = get_overlap(calc,
                               bands,
                               np.reshape(u1_nG, (Nn, Ng)), 
                               np.reshape(u2_nG, (Nn, Ng)),
                               calc.wfs.kpt_u[isk1].P_ani,
                               calc.wfs.kpt_u[isk2].P_ani,
                               bG_v) 
            indices = (ik1 + 1, ik2 + 1, G_c[0], G_c[1], G_c[2])
            print('%3d %3d %4d %3d %3d' % indices, file=f)
            for m1 in range(len(M_mm)):
                for m2 in range(len(M_mm)):
                    M = M_mm[m2, m1]
                    print('%20.12f %20.12f' % (M.real, M.imag), file=f)

    f.close()

def get_overlap(calc, bands, u1_nG, u2_nG, P1_ani, P2_ani, bG_v):

    Nn = len(u1_nG)
    M_nn = np.dot(u1_nG.conj(), u2_nG.T) * calc.wfs.gd.dv
    r_av = calc.atoms.positions / Bohr
    for ia in range(len(P1_ani.keys())):
        P1_ni = P1_ani[ia][bands]
        P2_ni = P2_ani[ia][bands]
        phase = np.exp(-1.0j * np.dot(bG_v, r_av[ia]))
        dO_ii = calc.wfs.setups[ia].dO_ii
        M_nn += P1_ni.conj().dot(dO_ii).dot(P2_ni.T) * phase

    return M_nn

def get_bands(seed):

    win_file = open(seed  + '.win')
    exclude_bands = None
    for line in win_file.readlines():
        l_e = line.split()
        if len(l_e) > 0:
            if l_e[0] == 'num_bands':
                Nn = int(l_e[2])
            if l_e[0] == 'exclude_bands':
                exclude_bands=line.split()[2]
                exclude_bands=[int(n) - 1 for n in exclude_bands.split(',')]
    if exclude_bands is None:
        bands = range(Nn)
    else:
        bands = range(Nn + len(exclude_bands))
        bands = [n for n in bands if n not in exclude_bands]
    win_file.close()

    return bands
