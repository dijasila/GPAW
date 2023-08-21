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
            s += f'{key}='
            if not isinstance(value, np.ndarray):
                print(value,'xxx')
                value = np.array(value)
            for val in value.ravel():
                print('val', val, type(val))
                if isinstance(val, np.bool_):
                    s += 'T' if val else 'F'
                else:
                    s+= str(val) 
                s += ','
            s += '\n'
        s += '/\n'
        return s

def call_libexexex(atoms, kpt, calc, hybrid_coeff=0.25):
    types = list(set(atoms.get_chemical_symbols()))
    species = np.array([ types.index(atom.symbol) for atom in atoms ]) 

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
        nml.add('REAL_EIGENVECTORS', False)
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
        nml.add('HSE_OMEGA_HF', 0.11)
        nml.add('CUTCB_RCUT', 7.1493418371540320)
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
        nml.add('USE_LOGSBT_FOR_RADIAL_HSE_INTEGRATION', True)
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
        lst = []
        basis = []
        cutoffs_S = {}
        for a, sphere in enumerate(calc.wfs.basis_functions.sphere_a):
            cutoffs = []
            S = species[a]
            nbasis = 0
            for j, spline in enumerate(sphere.spline_j):
                cutoffs.append(spline.get_cutoff())
                lst.extend( [j+1] * (2 * spline.l + 1) )
                nbasis += 2 * spline.l + 1
            basis.append(nbasis)
            cutoffs_S[S] = cutoffs
        nml.add('BASIS_FN', lst)
        nml.add('MAX_N_BASIS_SP',max(basis))
        nml.add('MAX_N_BASIS_SP2', 0)
        outer_radius = []
        for S in range(len(types)):
            outer_radius.extend(cutoffs_S[S])
        nml.add('OUTER_RADIUS', outer_radius)
        nml.add('SP2N_BASIS_SP', max(basis))
        #  BASIS_WAVE_SPL=  4.6603875251019841E-004,  5.711344187392619 <

        return str(nml)

    def grid():
        nml = NameList('EF_GRIDS') 
        ngrid = 1277
        ngridmin = 1.6666666245631252E-005
        ngridinc = 1.0123000144958496
        grid = ngridmin * ngridinc**np.arange(ngrid)
        nml.add('N_GRID', ngrid)
        nml.add('R_GRID_MIN', ngridmin)
        nml.add('R_GRID_INC', ngridinc)
        nml.add('R_GRID', grid)
     
        lst = []
        nradial = 34
        radius = 5.0 / Bohr
        scale_radial= -radius / np.log(1 - (nradial / (1 + nradial))**2)
        for s in range(len(species)):
            for i in range(1, nradial+1):
                r_scaled = i / (nradial + 1)
                lst.append(-np.log(1 - r_scaled**2) * scale_radial)
        nml.add('R_RADIAL', lst)          

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
 'ef_species_data.0000.nml':""" """,
 'ef_dimensions.0000.nml':""" """,
 'ef_localorb_io.0000.nml':""" """,
 'ef_physics.0000.nml':""" """,
 'ef_timing.0000.nml':""" """
 }

    for fname, data in templates.items():
        with open(fname, 'w') as f:
            print(data, file=f)
