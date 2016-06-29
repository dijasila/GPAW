"""GPAW I/O

Change log for version:

1) Initial version.

2) GridPoints array added when gpts is used.

3) Different k-points now have different number of plane-waves.  Added
   PlaneWaveIndices array.

4) Removed "UseSymmetry" and added "Symmetry" switches.

5) Added "ForcesConvergenceCriterion".

6) Added "ExternalPotential".

"""

import os
import warnings

from ase.units import AUT
from ase.units import Bohr, Hartree
from ase.data import atomic_names
from ase.atoms import Atoms
import numpy as np

import gpaw.mpi as mpi
from gpaw.io.dummy import DummyWriter


def open(filename, comm=mpi.world):
    if not filename.endswith('.gpw'):
        filename += '.gpw'
    import gpaw.io.tar as io

    # comm is used to eliminate replicated reads, hence it should never be None
    # it is not required for writes at the moment, but this is likely to change
    # in the future
    return io.Reader(filename, comm)


def write_atomic_matrix(writer, X_asp, name, master):
    all_X_asp = X_asp.deepcopy()
    all_X_asp.redistribute(all_X_asp.partition.as_serial())
    writer.add(name, ('nspins', 'nadm'), dtype=float)
    if master:
        writer.fill(all_X_asp.toarray(axis=1))

        
def write():
    
    w = open(filename, 'w', world)
    w['history'] = 'GPAW restart file'
    w['version'] = 6
    w['lengthunit'] = 'Bohr'
    w['energyunit'] = 'Hartree'

    # Write the k-points:
    if wfs.kd.N_c is not None:
        w.add('NBZKPoints', ('3'), wfs.kd.N_c, write=master)
        w.add('MonkhorstPackOffset', ('3'), wfs.kd.offset_c, write=master)
    w.dimension('nbzkpts', len(wfs.kd.bzk_kc))
    w.dimension('nibzkpts', len(wfs.kd.ibzk_kc))
    w.add('BZKPoints', ('nbzkpts', '3'), wfs.kd.bzk_kc, write=master)
    w.add('IBZKPoints', ('nibzkpts', '3'), wfs.kd.ibzk_kc, write=master)
    w.add('IBZKPointWeights', ('nibzkpts',), wfs.kd.weight_k, write=master)

    # Create dimensions for varioius netCDF variables:
    ng = wfs.gd.get_size_of_global_array()
    w.dimension('ngptsx', ng[0])
    w.dimension('ngptsy', ng[1])
    w.dimension('ngptsz', ng[2])
    ng = density.finegd.get_size_of_global_array()
    w.dimension('nfinegptsx', ng[0])
    w.dimension('nfinegptsy', ng[1])
    w.dimension('nfinegptsz', ng[2])
    w.dimension('nspins', wfs.nspins)
    w.dimension('nbands', wfs.bd.nbands)
    nproj = sum([setup.ni for setup in wfs.setups])
    nadm = sum([setup.ni * (setup.ni + 1) // 2 for setup in wfs.setups])
    w.dimension('nproj', nproj)
    w.dimension('nadm', nadm)

    if wfs.dtype == float:
        w['DataType'] = 'Float'
    else:
        w['DataType'] = 'Complex'

    p = paw.input_parameters

    if p.gpts is not None:
        w.add('GridPoints', ('3'), p.gpts, write=master)

    if p.h is None:
        w['GridSpacing'] = repr(None)
    else:
        w['GridSpacing'] = p.h / Bohr

    # Write various parameters:
    (w['KohnShamStencil'],
     w['InterpolationStencil']) = p['stencils']
    w['PoissonStencil'] = paw.hamiltonian.poisson.get_stencil()
    w['XCFunctional'] = paw.hamiltonian.xc.name
    w['Charge'] = p['charge']
    w['FixMagneticMoment'] = paw.occupations.fixmagmom
    w['SymmetryOnSwitch'] = wfs.kd.symmetry.point_group
    w['SymmetrySymmorphicSwitch'] = wfs.kd.symmetry.symmorphic
    w['SymmetryTimeReversalSwitch'] = wfs.kd.symmetry.time_reversal
    w['SymmetryToleranceCriterion'] = wfs.kd.symmetry.tol
    w['Converged'] = scf.converged
    w['FermiWidth'] = paw.occupations.width
    w['MixClass'] = density.mixer.__class__.__name__
    w['MixBeta'] = density.mixer.beta
    w['MixOld'] = density.mixer.nmaxold
    w['MixWeight'] = density.mixer.weight
    w['MaximumAngularMomentum'] = p.lmax
    w['SoftGauss'] = False
    w['FixDensity'] = p.fixdensity
    w['DensityConvergenceCriterion'] = p.convergence['density']
    w['EnergyConvergenceCriterion'] = p.convergence['energy'] / Hartree
    w['EigenstatesConvergenceCriterion'] = p.convergence['eigenstates']
    w['NumberOfBandsToConverge'] = p.convergence['bands']
    if p.convergence['forces'] is not None:
        force_unit = (Hartree / Bohr)
        w['ForcesConvergenceCriterion'] = p.convergence['forces'] / force_unit
    else:
        w['ForcesConvergenceCriterion'] = None
    w['Ekin'] = hamiltonian.Ekin
    w['e_coulomb'] = hamiltonian.e_coulomb
    w['Ebar'] = hamiltonian.Ebar
    w['Eext'] = hamiltonian.Eext
    w['Exc'] = hamiltonian.Exc
    w['S'] = hamiltonian.S
    try:
        if paw.occupations.fixmagmom:
            w['FermiLevel'] = paw.occupations.get_fermi_levels_mean()
            w['FermiSplit'] = paw.occupations.get_fermi_splitting()
        else:
            w['FermiLevel'] = paw.occupations.get_fermi_level()
    except ValueError:
        # Zero temperature calculation - don't write Fermi level:
        pass

    # write errors
    w['DensityError'] = scf.density_error
    w['EnergyError'] = scf.energy_error
    w['EigenstateError'] = scf.eigenstates_error

    # Try to write time and kick strength in time-propagation TDDFT:
    for attr, name in [('time', 'Time'), ('niter', 'TimeSteps'),
                       ('kick_strength', 'AbsorptionKick')]:
        if hasattr(paw, attr):
            value = getattr(paw, attr)
            if isinstance(value, np.ndarray):  # replicated write here
                w.add(name, ('3',), value, write=master)
            else:
                w[name] = value

    # Try to write FDTD-related data
    use_fdtd = hasattr(paw.hamiltonian.poisson, 'get_description') and \
        paw.hamiltonian.poisson.get_description() == 'FDTD+TDDFT'
    w['FDTD'] = use_fdtd
    if use_fdtd:
        paw.hamiltonian.poisson.write(paw, w)

    # Write fingerprint (md5-digest) for all setups:
    for setup in wfs.setups.setups.values():
        key = atomic_names[setup.Z] + 'Fingerprint'
        if setup.type != 'paw':
            key += '(%s)' % setup.type
        w[key] = setup.fingerprint

    setup_types = p['setups']
    if isinstance(setup_types, str):
        setup_types = {None: setup_types}
    for key, value in setup_types.items():
        if not isinstance(value, str):
            # Setups which are not strings are assumed to be
            # runtime-dependent and should *not* be saved.  We'll
            # just discard the whole dictionary
            setup_types = None
            break
    w['SetupTypes'] = repr(setup_types)

    basis = p['basis']  # And similarly for basis sets
    if isinstance(basis, dict):
        for key, value in basis.items():
            if not isinstance(value, str):
                basis = None
    w['BasisSet'] = repr(basis)

    dtype = {float: float, complex: complex}[wfs.dtype]

    timer.stop('Meta data')

    # Write projections:
    timer.start('Projections')
    w.add('Projections', ('nspins', 'nibzkpts', 'nbands', 'nproj'),
          dtype=dtype)
    if hdf5:
        # Domain masters write parallel over spin, kpoints and band groups
        all_P_ni = np.empty((wfs.bd.mynbands, nproj), dtype=wfs.dtype)
        cumproj_a = np.cumsum([0] + [setup.ni for setup in wfs.setups])
        for kpt in wfs.kpt_u:
            requests = []
            indices = [kpt.s, kpt.k]
            indices.append(wfs.bd.get_slice())
            do_write = (domain_comm.rank == 0)
            if domain_comm.rank == 0:
                P_ani = {}
                for a in range(natoms):
                    ni = wfs.setups[a].ni
                    if wfs.atom_partition.rank_a[a] == 0:
                        P_ani[a] = kpt.P_ani[a]
                    else:
                        P_ani[a] = np.empty((wfs.bd.mynbands, ni),
                                            dtype=wfs.dtype)
                        rank = wfs.atom_partition.rank_a[a]
                        requests.append(domain_comm.receive(P_ani[a],
                                                            rank,
                                                            1303 + a,
                                                            block=False))
            else:
                for a, P_ni in kpt.P_ani.items():
                    requests.append(domain_comm.send(P_ni, 0, 1303 + a,
                                                     block=False))
            domain_comm.waitall(requests)
            if domain_comm.rank == 0:
                for a in range(natoms):
                    all_P_ni[:, cumproj_a[a]:cumproj_a[a + 1]] = P_ani[a]
            w.fill(all_P_ni, parallel=parallel, write=do_write, *indices)
    else:
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                all_P_ni = wfs.collect_projections(k, s)
                if master:
                    w.fill(all_P_ni, s, k)
    
    del all_P_ni  # delete a potentially large matrix
    timer.stop('Projections')

    # Write atomic density matrices and non-local part of hamiltonian:
    timer.start('Atomic matrices')
    write_atomic_matrix(w, density.D_asp, 'AtomicDensityMatrices', master)
    write_atomic_matrix(w, hamiltonian.dH_asp, 'NonLocalPartOfHamiltonian',
                        master)
    timer.stop('Atomic matrices')

    # Write the eigenvalues and occupation numbers:
    timer.start('Band energies')
    for name, var in [('Eigenvalues', 'eps_n'), ('OccupationNumbers', 'f_n')]:
        w.add(name, ('nspins', 'nibzkpts', 'nbands'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                # if hdf5:  XXX Figure this out later
                #     indices = [s, k]
                #     indices.append(wfs.bd.get_slice())
                #     u = wfs.kd.where_is(s,k)
                #     a_nx = getattr(wfs.kpt_u[u], var)
                #     w.fill(a_nx, *indices, parallel=True)
                # else:
                a_n = wfs.collect_array(var, k, s)
                if master:
                    w.fill(a_n, s, k)
    timer.stop('Band energies')

    # Attempt to read the number of delta-scf orbitals:
    if hasattr(paw.occupations, 'norbitals'):
        norbitals = paw.occupations.norbitals
    else:
        norbitals = None

    # Write the linear expansion coefficients for Delta SCF:
    if mode == 'all' and norbitals is not None:
        timer.start('dSCF expansions')
        w.dimension('norbitals', norbitals)
        w.add('LinearExpansionOccupations', ('nspins',
              'nibzkpts', 'norbitals'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                ne_o = wfs.collect_auxiliary('ne_o', k, s, shape=norbitals)
                if master:
                    w.fill(ne_o, s, k)

        w.add('LinearExpansionCoefficients', ('nspins',
              'nibzkpts', 'norbitals', 'nbands'), dtype=complex)
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                for o in range(norbitals):
                    c_n = wfs.collect_array('c_on', k, s, subset=o)
                    if master:
                        w.fill(c_n, s, k, o)
        timer.stop('dSCF expansions')

    # Write the pseudopotential on the coarse grid:
    timer.start('Pseudo-potential')
    w.add('PseudoPotential',
          ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)

    for s in range(wfs.nspins):
        if hdf5:
            # XXXXXXX must work with density/hamiltonian redist changes
            # Maybe the easiest solution is to simply redist to wfs.gd
            do_write = (kpt_comm.rank == 0) and (band_comm.rank == 0)
            indices = [s] + hamiltonian.gd.get_slice()
            w.fill(hamiltonian.vt_sG[s], parallel=parallel, write=do_write,
                   *indices)
        else:
            vt_sG = hamiltonian.gd.collect(hamiltonian.vt_sG[s])
            if master:
                w.fill(vt_sG, s)
    timer.stop('Pseudo-potential')

    hamiltonian.xc.write(w, natoms)

    if p.external is not None:
        p.external.write(w)
    
    if mode in ['', 'all']:
        timer.start('Pseudo-wavefunctions')
        wfs.write(w, write_wave_functions=(mode == 'all'))
        timer.stop('Pseudo-wavefunctions')
    elif mode != '':
        w['Mode'] = 'fd'
        # Write the wave functions as separate files

        # check if we need subdirs and have to create them
        ftype, template = wave_function_name_template(mode)
        dirname = os.path.dirname(template)
        if dirname:
            if master and not os.path.isdir(dirname):
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                else:
                    raise RuntimeError("Can't create subdir " + dirname)
        else:
            dirname = '.'
        # the slaves have to wait until the directory is created
        world.barrier()
        paw.text('Writing wave functions to', dirname, 'using mode=', mode)

        ngd = wfs.gd.get_size_of_global_array()
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                for n in range(wfs.bd.nbands):
                    psit_G = wfs.get_wave_function_array(n, k, s)
                    if master:
                        fname = template % (s, k, n) + '.' + ftype
                        wpsi = open(fname, 'w')
                        wpsi.dimension('1', 1)
                        wpsi.dimension('ngptsx', ngd[0])
                        wpsi.dimension('ngptsy', ngd[1])
                        wpsi.dimension('ngptsz', ngd[2])
                        wpsi.add('PseudoWaveFunction',
                                 ('1', 'ngptsx', 'ngptsy', 'ngptsz'),
                                 dtype=dtype)
                        wpsi.fill(psit_G)
                        wpsi.close()


def read(paw, reader, read_projections=True):
    r = reader
    timer = paw.timer
    timer.start('Read')

    wfs = paw.wfs
    density = paw.density
    hamiltonian = paw.hamiltonian
    natoms = len(paw.atoms)

    world = paw.wfs.world
    gd = wfs.gd
    kd = wfs.kd
    bd = wfs.bd

    master = (world.rank == 0)
    parallel = (world.size > 1)

    version = r['version']

    hdf5 = hasattr(r, 'hdf5')

    # Verify setup fingerprints and count projectors and atomic matrices:
    for setup in wfs.setups.setups.values():
        try:
            key = atomic_names[setup.Z] + 'Fingerprint'
            if setup.type != 'paw':
                key += '(%s)' % setup.type
            if setup.fingerprint != r[key]:
                str = 'Setup for %s (%s) not compatible with restart file.' \
                    % (setup.symbol, setup.filename)
                if paw.input_parameters['idiotproof']:
                    raise RuntimeError(str)
                else:
                    warnings.warn(str)
        except (AttributeError, KeyError):
            str = 'Fingerprint of setup for %s (%s) not in restart file.' \
                % (setup.symbol, setup.filename)
            if paw.input_parameters['idiotproof']:
                raise RuntimeError(str)
            else:
                warnings.warn(str)
    nproj = sum([setup.ni for setup in wfs.setups])
    nadm = sum([setup.ni * (setup.ni + 1) // 2 for setup in wfs.setups])

    # Verify dimensions for minimally required netCDF variables:
    ng = gd.get_size_of_global_array()
    shapes = {'ngptsx': ng[0],
              'ngptsy': ng[1],
              'ngptsz': ng[2],
              'nspins': wfs.nspins,
              'nproj': nproj,
              'nadm': nadm}
    for name, dim in shapes.items():
        if r.dimension(name) != dim:
            raise ValueError('shape mismatch: expected %s=%d' % (name, dim))

    timer.start('Density')
    density.read(r, parallel, wfs.kptband_comm)
    timer.stop('Density')

    timer.start('Hamiltonian')
    hamiltonian.read(r, parallel)
    timer.stop('Hamiltonian')

    from gpaw.utilities.partition import AtomPartition
    atom_partition = AtomPartition(gd.comm, np.zeros(natoms, dtype=int))
    # <sarcasm>let's set some variables directly on some objects!</sarcasm>
    wfs.atom_partition = atom_partition
    density.atom_partition = atom_partition
    hamiltonian.atom_partition = atom_partition

    if version > 0.3:
        Etot = hamiltonian.Etot
        energy_error = r['EnergyError']
        if energy_error is not None:
            paw.scf.energies = [Etot, Etot + energy_error, Etot]
        wfs.eigensolver.error = r['EigenstateError']
        if version < 1:
            wfs.eigensolver.error *= gd.dv
    else:
        paw.scf.converged = r['Converged']

    if version > 0.6:
        if paw.occupations.fixmagmom:
            if 'FermiLevel' in r.get_parameters():
                paw.occupations.set_fermi_levels_mean(r['FermiLevel'])
            if 'FermiSplit' in r.get_parameters():
                paw.occupations.set_fermi_splitting(r['FermiSplit'])
        else:
            if 'FermiLevel' in r.get_parameters():
                paw.occupations.set_fermi_level(r['FermiLevel'])
    else:
        if (not paw.input_parameters.fixmom and
            'FermiLevel' in r.get_parameters()):
            paw.occupations.set_fermi_level(r['FermiLevel'])

    # Try to read the current time and kick strength in time-propagation TDDFT:
    for attr, name in [('time', 'Time'), ('niter', 'TimeSteps'),
                       ('kick_strength', 'AbsorptionKick')]:
        if hasattr(paw, attr):
            try:
                if r.has_array(name):
                    value = r.get(name, read=master)
                else:
                    value = r[name]
                setattr(paw, attr, value)
            except KeyError:
                pass

    # Try to read FDTD-related data
    try:
        use_fdtd = r['FDTD']
    except:
        use_fdtd = False

    if use_fdtd:
        from gpaw.fdtd.poisson_fdtd import FDTDPoissonSolver
        # fdtd_poisson will overwrite the poisson at a later stage
        paw.hamiltonian.fdtd_poisson = FDTDPoissonSolver(restart_reader=r,
                                                         paw=paw)

    # Try to read the number of Delta SCF orbitals
    try:
        norbitals = r.dimension('norbitals')
        paw.occupations.norbitals = norbitals
    except (AttributeError, KeyError):
        norbitals = None

    nibzkpts = r.dimension('nibzkpts')
    nbands = r.dimension('nbands')
    nslice = bd.get_slice()

    if (nibzkpts != len(wfs.kd.ibzk_kc) or
        nbands != bd.comm.size * bd.mynbands):
        paw.scf.reset()
    else:
        # Verify that symmetries for for k-point reduction hasn't changed:
        tol = 1e-12

        if master:
            bzk_kc = r.get('BZKPoints', read=master)
            weight_k = r.get('IBZKPointWeights', read=master)
            assert np.abs(bzk_kc - kd.bzk_kc).max() < tol
            assert np.abs(weight_k - kd.weight_k).max() < tol

        for kpt in wfs.kpt_u:
            # Eigenvalues and occupation numbers:
            timer.start('Band energies')
            k = kpt.k
            s = kpt.s
            if hdf5:  # fully parallelized over spins, k-points
                do_read = (gd.comm.rank == 0)
                indices = [s, k]
                indices.append(nslice)
                kpt.eps_n = r.get('Eigenvalues', parallel=parallel,
                                  read=do_read, *indices)
                gd.comm.broadcast(kpt.eps_n, 0)
                kpt.f_n = r.get('OccupationNumbers', parallel=parallel,
                                read=do_read, *indices)
                gd.comm.broadcast(kpt.f_n, 0)
            else:
                eps_n = r.get('Eigenvalues', s, k, read=master)
                f_n = r.get('OccupationNumbers', s, k, read=master)
                kpt.eps_n = eps_n[nslice].copy()
                kpt.f_n = f_n[nslice].copy()
            timer.stop('Band energies')

            if norbitals is not None:  # XXX will probably fail for hdf5
                timer.start('dSCF expansions')
                kpt.ne_o = np.empty(norbitals, dtype=float)
                kpt.c_on = np.empty((norbitals, bd.mynbands), dtype=complex)
                for o in range(norbitals):
                    kpt.ne_o[o] = r.get('LinearExpansionOccupations', s, k, o,
                                        read=master)
                    c_n = r.get('LinearExpansionCoefficients', s, k, o,
                                read=master)
                    kpt.c_on[o, :] = c_n[nslice]
                timer.stop('dSCF expansions')

        if (r.has_array('PseudoWaveFunctions') and
            paw.input_parameters.mode != 'lcao'):

            timer.start('Pseudo-wavefunctions')
            wfs.read(r, hdf5)
            timer.stop('Pseudo-wavefunctions')

        if (r.has_array('WaveFunctionCoefficients') and
            paw.input_parameters.mode == 'lcao'):
            wfs.read_coefficients(r)

        timer.start('Projections')
        if hdf5 and read_projections:
            # Domain masters read parallel over spin, kpoints and band groups
            cumproj_a = np.cumsum([0] + [setup.ni for setup in wfs.setups])
            all_P_ni = np.empty((bd.mynbands, cumproj_a[-1]),
                                dtype=wfs.dtype)
            for kpt in wfs.kpt_u:
                kpt.P_ani = {}
                indices = [kpt.s, kpt.k]
                indices.append(bd.get_slice())
                do_read = (gd.comm.rank == 0)
                # timer.start('ProjectionsCritical(s=%d,k=%d)' % (kpt.s,kpt.k))
                r.get('Projections', out=all_P_ni, parallel=parallel,
                      read=do_read, *indices)
                # timer.stop('ProjectionsCritical(s=%d,k=%d)' % (kpt.s,kpt.k))
                if gd.comm.rank == 0:
                    for a in range(natoms):
                        ni = wfs.setups[a].ni
                        P_ni = np.empty((bd.mynbands, ni), dtype=wfs.dtype)
                        P_ni[:] = all_P_ni[:, cumproj_a[a]:cumproj_a[a + 1]]
                        kpt.P_ani[a] = P_ni

            del all_P_ni  # delete a potentially large matrix
        elif read_projections and r.has_array('Projections'):
            wfs.read_projections(r)
        timer.stop('Projections')

    # Get the forces from the old calculation:
    if r.has_array('CartesianForces'):
        paw.forces.F_av = r.get('CartesianForces', broadcast=True)
    else:
        paw.forces.reset()

    # Manage mode change:
    paw.scf.check_convergence(density, wfs.eigensolver, wfs, hamiltonian,
                              paw.forces)
    newmode = paw.input_parameters.mode
    try:
        oldmode = r['Mode']
        if oldmode == 'pw':
            from gpaw.wavefunctions.pw import PW
            oldmode = PW(ecut=r['PlaneWaveCutoff'] * Hartree)
    except (AttributeError, KeyError):
        oldmode = 'fd'  # This is an old gpw file from before lcao existed
        
    spos_ac = paw.atoms.get_scaled_positions() % 1.0
    if newmode == 'lcao':
        wfs.load_lazily(hamiltonian, spos_ac)

    rank_a = density.gd.get_ranks_from_positions(spos_ac)
    new_atom_partition = AtomPartition(density.gd.comm, rank_a)
    for obj in [density, hamiltonian]:
        obj.set_positions_without_ruining_everything(spos_ac,
                                                     new_atom_partition)

    if newmode != oldmode:
        paw.scf.reset()

    hamiltonian.xc.read(r)

    timer.stop('Read')


def read_atoms(reader):

    positions = reader.get('CartesianPositions', broadcast=True) * Bohr
    numbers = reader.get('AtomicNumbers', broadcast=True)
    cell = reader.get('UnitCell', broadcast=True) * Bohr
    pbc = reader.get('BoundaryConditions', broadcast=True)
    tags = reader.get('Tags', broadcast=True)
    magmoms = reader.get('MagneticMoments', broadcast=True)

    # Create instance of Atoms object, and set_tags and magnetic moments
    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  cell=cell,
                  pbc=pbc)

    if tags.any():
        atoms.set_tags(tags)
    if magmoms.any():
        atoms.set_initial_magnetic_moments(magmoms)

    if reader.has_array('CartesianVelocities'):
        velocities = reader.get('CartesianVelocities',
                                broadcast=True) * Bohr / AUT
        atoms.set_velocities(velocities)

    return atoms


class FileReference:
    """Common base class for having reference to a file. The actual I/O
       classes implementing the referencing should be inherited from
       this class."""

    def __init__(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def __len__(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def __array__(self):
        return self[::]
