def write_win(overlaps, **kwargs):
    f = open(seed + '.win', 'w')
    print('num_bands       = %d' % len(bands), file=f)
    print('num_wann        = %d' % Nw, file=f)
    print('num_iter        = %d' % num_iter, file=f)
    print(file=f)

    if len(bands) > Nw:
        ef = calc.get_fermi_level()
        print('fermi_energy  = %2.3f' % ef, file=f)
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



def write_overlaps(calc, seed=None, spin=0, soc=None, less_memory=False):

    if seed is None:
        seed = calc.atoms.get_chemical_formula()

    if soc is None:
        spinors = False
    else:
        spinors = True

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
        return soc[bz_index].wavefunctions(
            calc, periodic=True)[bands]

    if not less_memory:
        u_knG = []
        for ik in range(Nk):
            if spinors:
                # For spinors, G denotes spin and grid: G = (s, gx, gy, gz)
                u_nG = wavefunctions(ik)
                u_knG.append(u_nG[bands])
            else:
                # For non-spinors, G denotes grid: G = (gx, gy, gz)
                u_knG.append(np.array([wfs.get_wave_function_array(n, ik, spin)
                                       for n in bands]))

    P_kani = []
    for ik in range(Nk):
        if spinors:
            P_kani.append(soc[ik].P_amj)
        else:
            P_kani.append(calc.wfs.kpt_qs[ik][spin].P_ani)

    for ik1 in range(Nk):
        if less_memory:
            if spinors:
                u1_nG = wavefunctions(ik1)
            else:
                u1_nG = np.array([wfs.get_wave_function_array(n, ik1, spin)
                                  for n in bands])
        else:
            u1_nG = u_knG[ik1]
        for ib in range(Nb):
            # b denotes nearest neighbor k-points
            line = lines[i0 + ik1 * Nb + ib].split()
            ik2 = int(line[1]) - 1
            if less_memory:
                if spinors:
                    u2_nG = wavefunctions(ik2)
                else:
                    u2_nG = np.array([wfs.get_wave_function_array(n, ik2, spin)
                                      for n in bands])
            else:
                u2_nG = u_knG[ik2]

            G_c = np.array([int(line[i]) for i in range(2, 5)])
            bG_c = kpts_kc[ik2] - kpts_kc[ik1] + G_c
            bG_v = np.dot(bG_c, icell_cv)
            u2_nG = u2_nG * np.exp(-1.0j * gemmdot(bG_v, r_g, beta=0.0))
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
