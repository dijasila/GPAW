from ase.units import Bohr  # TODO: Import from CODATA 2022
import numpy as np


class NameList:
    def __init__(self, title):
        self.title = title
        self.lines = []

    def add(self, arrayname, values):
        self.lines.append((arrayname, values)) 

    def __str__(self):
        s = f'&{self.title}\n'
        for key, value in self.lines:
            s += f' {key}='
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            print(key, value.shape)
            if len(value.ravel())> 0 and np.iscomplex(value.ravel()[0]):
                for val in value.ravel():
                    s+= f'({val.real},{val.imag})' 
                    s += ','
            else:
                for val in value.ravel():
                    if isinstance(val, np.bool_):
                        s += 'T' if val else 'F'
                    elif isinstance(val, np.str_):
                        s += f'"{val}"'
                    else:
                        s+= str(val) 
                    s += ','
            s += '\n'
        s += '/\n'
        return s

def call_libexexex(atoms, kpt, calc,  xc=None):
    if xc == 'PBE0':
        hybrid_coeff = 0.25
        hse_omega_hf = 0.0
        use_logsbt_for_radial_hse_integration = False
        use_hse = False
    elif xc == 'HSE06':
        hybrid_coeff = 0.25
        hse_omega_hf = 0.11
        use_hse = True
        use_logsbt_for_radial_hse_integration = True
    else:
        raise ValueError('Only PBE0 and HSE06 supported')

    types = list(set(atoms.get_chemical_symbols()))
    species = np.array([ types.index(atom.symbol) for atom in atoms ]) 
    nspecies = len(types)
    n_max_ind_fns = 2
    n_max_grid = 1187
    ext_l_shell_max = 1
    nstates = len(calc.wfs.kpt_u[0].C_nM)
    nbasis = calc.wfs.kpt_u[0].C_nM.shape[1]
    nradial = 167 #34
    n_kpoints_nosym = len(calc.wfs.kd.bzk_kc)
    lst = []
    basisfn = []
    cutoffs_S = {}
    ls = []
    fn_species = []
    basis_l = []
    basis_m = []
    basis_atom = []
    ext_fn = []
    basis_fn_species = []
    J = 0
    max_fn_per_l = 0
    lsp2basis_fn = np.zeros((5, 5, nspecies), dtype=int)
    lsp2n_basis_fnlsp = np.zeros((max(basis_l) + 1, nspecies), dtype=int)
    for a, sphere in enumerate(calc.wfs.basis_functions.sphere_a):
        cutoffs = []
        S = species[a]
        nbasis = 0
        max_fn_per_l = max(max_fn_per_l, max([len([None for spline in sphere.spline_j if spline.l == lp]) for lp in range(4)]))
        basis_fn_species.append(len(sphere.spline_j))
        numperl = [0] * 5
        for j, spline in enumerate(sphere.spline_j):
            lsp2basis_fn[numperl[spline.l], spline.l, S] = J + 1
            numperl[spline.l] += 1
            ls.append(spline.l)
            fn_species.append(a + 1)
            cutoffs.append(spline.get_cutoff())
            lst.extend( [j+1] * (2 * spline.l + 1) )
            nbasis += 2 * spline.l + 1
            for m in range(-spline.l, spline.l + 1):
                ext_fn.append(J + 1)
                basis_l.append(spline.l)
                basis_m.append(m)
                basis_atom.append(a + 1)
            J += 1
        basisfn.append(nbasis)
        cutoffs_S[S] = cutoffs
        lsp2basis_fn = lsp2basis_fn[:max_fn_per_l, :max(basis_l) + 1]

    def geometry_nml():
        nml = NameList('EF_GEOMETRY')
        nml.add('LATTICE_VECTOR', atoms.cell / Bohr)
        nml.add('RECIP_LATTICE_VECTOR', np.linalg.inv(atoms.cell).T * 2 * np.pi * Bohr)
        nml.add('COORDS', atoms.get_positions() / Bohr)
        nml.add('SPECIES', species + 1)
        nml.add('FRAC_COORDS', atoms.get_scaled_positions())
        nml.add('EMPTY', np.array([False] * len(atoms)))
        return str(nml)

    def runtime_choices():
        nml = NameList('EF_RUNTIME_CHOICES')
        nml.add('USE_SCALAPACK', False)
        nml.add('REAL_EIGENVECTORS', calc.wfs.dtype != complex)
        nml.add('USE_FULL_SYMMETRY', False)
        nml.add('USE_SYMMETRY_REDUCED_K_GRID', True)
        nml.add('USE_SYMMETRY_REDUCED_SPG', False)
        nml.add('N_K_POINTS_XYZ', kpt)
        nml.add('N_K_POINTS_XYZ_NOSYM', kpt)
        nml.add('K_POINTS_OFFSET', np.array([0.0, 0.0, 0.0]))
        nml.add('WAVE_THRESHOLD', 1e-6)
        nml.add('HYBRID_COEFF', hybrid_coeff)
        nml.add('COUL_MAT_THRESHOLD', 1.0000000000000000E-010)
        nml.add('CRIT_VAL', 1e-7)
        nml.add('FLAG_SPLIT_ATOMS', False)
        nml.add('FLAG_SPLIT_BATCH', False)
        nml.add('FLAG_SPLIT_MAX_VAL', False)
        nml.add('FOCK_MATRIX_BLOCKING', -1)
        nml.add('FOCK_MATRIX_LOAD_BALANCE', False)
        nml.add('GET_FULL_DENSITY_FIRST', True)
        nml.add('AS_COMPONENTS', 6)
        nml.add('HSE_OMEGA_HF', hse_omega_hf)
        nml.add('CUTCB_RCUT', (atoms.cell.volume / Bohr**3 * n_kpoints_nosym * 3 / (4 * np.pi))**(1/3))
        nml.add('CUTCB_WIDTH', 1.0918686923587797)
        nml.add('LRC_PT2_STARTED', False)
        nml.add('FLAG_AUXIL_BASIS', 1)
        nml.add('SBTGRID_N', 4096)
        nml.add('SBTGRID_LNK0', -25.000000000000000)
        nml.add('SBTGRID_LNR0', -38.000000000000000)
        nml.add('SBTGRID_LNRANGE', 45.000000000000000)
        nml.add('PRODBAS_NB', 0)
        nml.add('ERS_WIDTH', 1.0000000000000000)
        nml.add('ERS_OMEGA', 1.0000000000000000)
        nml.add('ERS_RCUT', 0.52917720999999995)
        nml.add('ERFC_OMEGA', 1.0000000000000000)
        nml.add('USE_SPG_FULL_DELTA', True)
        nml.add('USE_K_PHASE_EXX', True)
        nml.add('FLAG_SPLIT_MIN_VAL', False)
        nml.add('SPLIT_MIN_VAL', 1)
        nml.add('SPLIT_MAX_VAL',1)
        nml.add('SPLIT_BATCH', 1.0000000000000000)
        nml.add('FOCK_MATRIX_NODES_PER_INSTANCE', 10000000)
        nml.add('FOCK_MATRIX_INSTANCES_PER_NODE', 10000000)
        nml.add('FOCK_MATRIX_MAX_MEM_PER_NODE', -1.0000000000000000)
        nml.add('LC_DIELECTRIC_CONSTANT', 1.0000000000000000)
        nml.add('OVLP_MAT_THRESHOLD', 1.0000000000000000E-010)
        nml.add('USE_LC_REFERENCE', True)
        nml.add('PRODBAS_THRESHOLD', 1.0000000000000001E-005)
        nml.add('SAFE_MINIMUM', 2.2250738585072014E-308)
        nml.add('ADAMS_MOULTON_INTEGRATOR', True)
        nml.add('FLAG_BASIS', 0)
        nml.add('RI_TYPE', 3)
        nml.add('USE_LOGSBT_FOR_RADIAL_HSE_INTEGRATION', use_logsbt_for_radial_hse_integration)
        nml.add('FLAG_REL', 2)
        nml.add('MIXER', 1)
        nml.add('CHARGE_MIX_PARAM', 5.0000000745058060E-002)
        nml.add('LINEAR_MIX_PARAM', 0.0000000000000000)
        nml.add('FLAG_PERIODIC_GW_OPTIMIZE_INIT_FFT', False)
        nml.add('FLAG_PERIODIC_GW_OPTIMIZE_INIT_SCALE', False)
        nml.add('OUT_BASIS', False)
        nml.add('USE_ELSI_DM', False)
        nml.add('ELSI_CAN_READ_DM', False)
        nml.add('ELSI_READ_DM_DONE_HF', False)
        nml.add('SC_ITER_LIMIT', 1000)
        nml.add('PERIODIC_GW_OPTIMIZE_KGRID_SYMMETRY', 1)
        nml.add('OCCUPATION_TYPE', 0)
        nml.add('CALCULATE_ATOM_BSSE', False)
        nml.add('FLAG_PERIODIC_GW_OPTIMIZE_INIT_INVERSE', False)
        nml.add('SYMMETRY_THRESH', 1.0000000000000000E-004)
        return str(nml)

    def basis():
        nml = NameList('EF_BASIS')
        nml.add('ATOM2BASIS_OFF', calc.wfs.basis_functions.M_a)
        nml.add('BASIS_FN', lst)
        nml.add('MAX_N_BASIS_SP',max(basisfn))
        nml.add('MAX_N_BASIS_SP2', 0)
        outer_radius = []
        for S in range(len(types)):
            outer_radius.extend(cutoffs_S[S])
        nml.add('OUTER_RADIUS', outer_radius)
        nml.add('SP2N_BASIS_SP', max(basisfn))
        nml.add('BASIS_WAVE_SPL', 0) #=  4.6603875251019841E-004,  5.711344187392619 <

        """
        Data content mismatch ef_basis.0000.nml basisfn_l data: 0 refdata: [0, 1]
        Data content mismatch ef_basis.0000.nml basisfn_species data: 0 refdata: [1, 1]
        Data content mismatch ef_basis.0000.nml lsp2basis_fn data: 0 refdata: [1, 2]
        Data content mismatch ef_basis.0000.nml lsp2n_basis_fnlsp data: 0 refdata: [1, 1]
        Data content mismatch ef_basis.0000.nml basis_l data: 0 refdata: [0, 1, 1, 1]
        Data content mismatch ef_basis.0000.nml basis_m data: 0 refdata: [0, -1, 0, 1]
        Data content mismatch ef_basis.0000.nml lsp2basis_sp data: 0 refdata: [1, 3]
        Data content mismatch ef_basis.0000.nml basis_atom data: 0 refdata: [1, 1, 1, 1]
        Data content mismatch ef_basis.0000.nml basis_mapping data: 0 refdata: [0, 0, 0, 0]
        Data content mismatch ef_basis.0000.nml n_basis_fn_species data: 0 refdata: 2
        Data content mismatch ef_basis.0000.nml atom_radius data: 0 refdata: 7.787708969805111
        Data content mismatch ef_basis.0000.nml ext_atom data: 0 refdata: [1, 1, 1, 1]
        Data content mismatch ef_basis.0000.nml n_ext_fn_species data: 0 refdata: 2
        Data content mismatch ef_basis.0000.nml ext_fn data: 0 refdata: [1, 2, 2, 2]
        Data content mismatch ef_basis.0000.nml ext_l data: 0 refdata: [0, 1, 1, 1]
        Data content mismatch ef_basis.0000.nml ext_m data: 0 refdata: [0, -1, 0, 1]
        Data content mismatch ef_basis.0000.nml extfn_n data: 0 refdata: [0, 1]
        """
        nml.add('BASISFN_L', ls) #=0          ,1          ,
        nml.add('BASISFN_SPECIES', fn_species) #= 2*1          ,
        nml.add('LSP2BASIS_FN', lsp2basis_fn) # =1          ,2          ,
        nml.add('LSP2N_BASIS_FNLSP', lsp2n_basis_fnlsp) #= 2*1          ,
        nml.add('BASIS_L', basis_l) # =0          , 3*1          ,
        nml.add('BASIS_M', basis_m) #=0          ,-1         ,0          ,1          ,
        nml.add('LSP2BASIS_SP', 0) #=1          ,3          ,
        nml.add('BASIS_ATOM', basis_atom)#= 4*1          ,
        nml.add('BASIS_MAPPING', [0] * len(basis_l)) #= 4*0          ,
        nml.add('N_BASIS_FN_SPECIES', basis_fn_species) #=2          ,
        nml.add('ATOM_RADIUS', 0) #=  7.7877089698051112     ,
        nml.add('EXT_ATOM', basis_atom) #= 4*1          ,
        nml.add('N_EXT_FN_SPECIES', 0) #=2          ,
        nml.add('EXT_FN', ext_fn) #=1          , 3*2          ,
        nml.add('EXT_L', basis_l) #=0          , 3*1          ,
        nml.add('EXT_M', basis_m) # =0          ,-1         ,0          ,1          ,
        nml.add('EXTFN_N', 0) #=0          ,1          ,
        nml.add('EXT_WAVE_SPL', 0) #=  5.0899054203602369E-005,  6.2443600775498648E-007, -1.0587911840678754E-022,  1.6230958255270334E-009,  5.1525113307182883E-005,

        return str(nml)

    def grid():
        nml = NameList('EF_GRIDS') 
        ngrid = n_max_grid
        ngridmin = 4.999999873689376e-05
        #ngridmin = 1.6666666245631252E-005
        ngridinc = 1.0123000144958496
        grid = ngridmin * ngridinc**np.arange(ngrid)
        nml.add('N_GRID', ngrid)
        nml.add('R_GRID_MIN', ngridmin)
        nml.add('R_GRID_INC', ngridinc)
        nml.add('R_GRID', grid)
     
        lst = []
        radius = 5.0 / Bohr
        scale_radial= -radius / np.log(1 - (nradial / (1 + nradial))**2)
        for s in range(len(types)):
            for i in range(1, nradial+1):
                r_scaled = i / (nradial + 1)
                lst.append(-np.log(1 - r_scaled**2) * scale_radial)
        nml.add('R_RADIAL', lst)          

        return str(nml)

    def speciesdata():
        nml = NameList('EF_SPECIES_DATA') 
        nml.add('NO_BASIS', [False] * nspecies)
        nml.add('SPECIES_PSEUDOIZED', [False] * nspecies)
        nml.add('SPECIES_NAME', types)
        nml.add('INCLUDE_MIN_BASIS', [False] * nspecies)
        nml.add('EXT_L_SHELL_MAX', [ext_l_shell_max] * nspecies)
        nml.add('INNERMOST_MAX', [2] * nspecies)
        nml.add('N_ATOMIC', [0] * nspecies)
        nml.add('ATOMS_IN_STRUCTURE', [ len([None for atom in atoms if atom.symbol == symbol]) for symbol in types])
        nml.add('MAX_L_PRODBAS', [20] * nspecies)
        nml.add('N_AUX_GAUSSIAN', [0] * nspecies)
        nml.add('ATOMIC_L', [0] * (nspecies * n_max_ind_fns))
        nml.add('PRODBAS_ACC', [1e-4] * nspecies)
        nml.add('ATOMIC_WAVE', [0] * ( n_max_grid * nspecies * n_max_ind_fns ))
        return str(nml)

    def dimensions():
        nml = NameList('EF_DIMENSIONS')
        nml.add('N_PERIODIC', 3)
        nml.add('N_ATOMS', len(atoms))
        nml.add('N_SPECIES', len(types))
        nml.add('N_CENTERS', 27)
        nml.add('N_SPIN', 1)
        nml.add('N_STATES', nstates)

        nml.add('N_K_POINTS', len(calc.wfs.kd.ibzk_kc))
        nml.add('N_K_POINTS_NOSYM', n_kpoints_nosym)
        nml.add('N_K_POINTS_TASK', len(calc.wfs.kd.ibzk_kc))
        nml.add('N_IRK_POINTS', 1)
        nml.add('N_IRK_POINTS_TASK', 1)
        nml.add('N_BASIS', nbasis)
        nml.add('N_BASIS_FNS', 2)
        nml.add('N_CENTERS_BASIS_T', 4)
        nml.add('N_BASBAS',0)
        nml.add('N_BASBAS_FNS',0)
        nml.add('N_BASBAS_SUPERCELL', 0)
        nml.add('N_LOC_PRODBAS', 0)
        nml.add('N_LOC_PRODBAS_SUPERCELL', 0)
        nml.add('N_MAX_LOC_PRODBAS', 0)
        nml.add('N_MAX_LOC_PRODBAS_SUPERCELL', 0)
        nml.add('L_WAVE_MAX', ext_l_shell_max)
        nml.add('N_EXT', nbasis)
        nml.add('N_MAX_BASIS_FNS', 2)
        nml.add('N_HARTREE_GRID', n_max_grid)
        nml.add('N_MAX_GRID', n_max_grid)
        nml.add('N_MAX_COEFF_3FN', 16)
        nml.add('N_MAX_SPLINE', 4)
        nml.add('N_MAX_PULAY', 8)
        nml.add('USE_LC_WPBEH', False)
        nml.add('FLAG_MAX_COEFF_3FN', False)
        nml.add('USE_RPA_FORCE', False)
        nml.add('USE_ERFC', False)
        nml.add('USE_ERS', False)
        nml.add('USE_GW', False)
        nml.add('USE_GW_AND_HSE', False)
        nml.add('USE_HSE', use_hse)
        nml.add('LEAST_MEMORY_3', False)
        nml.add('USE_CUTCB', True)
        nml.add('EXCLUDE_GZERO', False)
        nml.add('USE_LOGSBT', True)
        nml.add('USE_DFTPT2_AND_HSE', False)
        nml.add('USE_THREADSAFE_GWINIT', False)
        nml.add('N_EXT_FNS', 6)
        nml.add('MAX_N_BASIS_FNLSP', max_fn_per_l)
        nml.add('MAX_BASIS_L', max(basis_l))
        nml.add('N_MAX_RADIAL', 34)
        nml.add('N_CELLS_BVK', 1)
        nml.add('N_MAX_IND_FNS', n_max_ind_fns)
        nml.add('N_CELLS_PAIRS', 27)
        nml.add('N_CENTERS_ELE_SUMMATION',1)
        return str(nml)

    def timing():
        nml = NameList('EF_TIMING')
        nml.add('TOT_CLOCK_TIME_MATRIX_IO', 0.0)
        nml.add('TOT_TIME_MATRIX_IO', 0.0)
        return str(nml)

    def physics():
        nml = NameList('EF_PHYSICS')
        C = []
        f = []
        for kpt in calc.wfs.kpt_u:
            C.extend(kpt.C_nM.ravel())
            f.extend(kpt.f_n.ravel())
        if np.iscomplex(kpt.C_nM[0,0]):
            nml.add('KS_EIGENVECTOR_COMPLEX', C)
            nml.add('KS_EIGENVECTOR', 0.0000000000000000)
        else:
            nml.add('KS_EIGENVECTOR_COMPLEX', 0.0)
            nml.add('KS_EIGENVECTOR', C)

        nml.add('OCC_NUMBERS', f)
        nml.add('EIGENVEC', 0)
        nml.add('EIGENVEC_COMPLEX', 0+0j)
        return str(nml)

    def localorb():
        nml = NameList('EF_LOCALORB_IO')
        nml.add('USE_UNIT', 6)
        nml.add('OUTPUT_PRIORITY', 0)
        return str(nml)

    templates = {'ef_analytical_stress.0000.nml':
"""&EF_ANALYTICAL_STRESS
 AS_EXX_STRESS_LOCAL= 9*0.0000000000000000       ,
 /""",

'ef_mpi_tasks.0000.nml':
"""&EF_MPI_TASKS
 USE_MPI_IN_PLACE=T,
 /""",
 'ef_localorb_io.0000.nml':
 """&EF_LOCALORB_IO
 USE_UNIT=6          ,
 OUTPUT_PRIORITY=0          ,
 /""",
 'ef_geometry.0000.nml': geometry_nml(),
 'ef_runtime_choices.0000.nml':runtime_choices(),
 'ef_basis.0000.nml':basis(),
 'ef_grids.0000.nml':grid(),
 'ef_pbc_lists.0000.nml':""" """,
 'ef_species_data.0000.nml':speciesdata(),
 'ef_dimensions.0000.nml':dimensions(),
 'ef_localorb_io.0000.nml':localorb(),
 'ef_physics.0000.nml':physics(),
 'ef_timing.0000.nml':timing()
 }

    for fname, data in templates.items():
        with open(fname, 'w') as f:
            print(data, file=f)

def verify(folder1, folder2):
    files = ['ef_analytical_stress.0000.nml',
             'ef_mpi_tasks.0000.nml',
             'ef_localorb_io.0000.nml',
             'ef_geometry.0000.nml',
             'ef_runtime_choices.0000.nml',
             'ef_basis.0000.nml',
             'ef_grids.0000.nml',
             'ef_pbc_lists.0000.nml',
             'ef_species_data.0000.nml',
             'ef_dimensions.0000.nml',
             'ef_localorb_io.0000.nml',
             'ef_physics.0000.nml',
             'ef_timing.0000.nml']
    import f90nml
    from pathlib import Path
    def compare(nml1, nml2, fname):
        errors = 0
        for key in nml2.keys():
            try:
                nml1[key]
            except KeyError:
                print('Missing name', key, 'in', fname)
                errors += 1
                continue
            if nml1[key] == nml2[key]:
                pass # print('Matching data', key)
            else:
                if hasattr(nml1[key], '__len__'):
                    if not hasattr(nml2[key], '__len__'):
                        print('Mismatch of types', key, nml1[key], nml2[key])
                        errors += 1
                        continue
                    if len(nml1[key]) != len(nml2[key]):
                        print('Data size mismatch', fname, key, 'size:', len(nml1[key]), 'refsize:', len(nml2[key]))
                        errors += 1
                    else:
                        if nml1[key] != nml2[key]:
                            if not np.allclose(np.array(nml1[key]), np.array(nml2[key])):
                                print('Data content mismatch', fname, key)
                                errors += 1
                                if len(nml1[key]) < 10:
                                    print('data:', nml1[key], 'refdata:', nml2[key])
                else:
                    if nml1[key] != nml2[key]:
                        print('Data content mismatch', fname, key, 'data:', nml1[key], 'refdata:', nml2[key])
                        errors += 1
        return errors

    errs = 0
    for fname in files:
        nmls = [f90nml.read(Path(folder) / fname) for folder in [folder1, folder2]]
        for key in nmls[1].keys():
            try:
                nmls[0][key]
            except KeyError:
                print('Missing namelist', fname, key)
                continue
            errs += compare(nmls[0][key], nmls[1][key], fname)
    print('-----------------------------')
    print('Total errors', errs)
