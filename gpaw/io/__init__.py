import os

from ASE.ChemicalElements.name import names
import Numeric as num

import gpaw.mpi as mpi


MASTER = 0

def open(filename, mode='r'):
    if filename.endswith('.nc'):
        import gpaw.io.netcdf as io
    else:
        if not filename.endswith('.gpw'):
            filename += '.gpw'
        import gpaw.io.tar as io

    if mode == 'r':
        return io.Reader(filename)
    elif mode == 'w':
        return io.Writer(filename)
    else:
        raise ValueError("Illegal mode!  Use 'r' or 'w'.")


def write(paw, filename, mode):
    if mpi.rank == MASTER:
        w = open(filename, 'w')

        w['history'] = 'GPAW restart file'
        w['version'] = '0.4'
        w['lengthunit'] = 'Bohr'
        w['energyunit'] = 'Hartree'

        pos_ac, Z_a, cell_cc, pbc_c = paw.last_atomic_configuration
        tag_a, magmom_a = paw.extra_list_of_atoms_stuff
        
        w.dimension('natoms', paw.natoms)
        w.dimension('3', 3)

        w.add('AtomicNumbers', ('natoms',), Z_a, units=(0, 0))
        w.add('CartesianPositions', ('natoms', '3'), pos_ac, units=(1, 0))
        w.add('MagneticMoments', ('natoms',), magmom_a, units=(0, 0))
        w.add('Tags', ('natoms',), tag_a, units=(0, 0))
        w.add('BoundaryConditions', ('3',), pbc_c, units=(0, 0))
        w.add('UnitCell', ('3', '3'), cell_cc, units=(1, 0))

        w.add('PotentialEnergy', (), paw.Etot + 0.5 * paw.S,
              units=(0, 1))
        if paw.F_ac is not None:
            w.add('CartesianForces', ('natoms', '3'), paw.F_ac, units=(-1, 1))
        
        # Write the k-points:
        w.dimension('nbzkpts', len(paw.bzk_kc))
        w.dimension('nibzkpts', paw.nkpts)
        w.add('BZKPoints', ('nbzkpts', '3'), paw.bzk_kc)
        w.add('IBZKPoints', ('nibzkpts', '3'), paw.ibzk_kc)
        w.add('IBZKPointWeights', ('nibzkpts',), paw.weight_k)

        # Create dimensions for varioius netCDF variables:
        ng = paw.gd.get_size_of_global_array()
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        ng = paw.finegd.get_size_of_global_array()
        w.dimension('nfinegptsx', ng[0])
        w.dimension('nfinegptsy', ng[1])
        w.dimension('nfinegptsz', ng[2])
        w.dimension('nspins', paw.nspins)
        w.dimension('nbands', paw.nbands)
        
        nproj = 0
        nadm = 0
        for nucleus in paw.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nproj += ni
            nadm += ni * (ni + 1) / 2
        w.dimension('nproj', nproj)
        w.dimension('nadm', nadm)

        p = paw.input_parameters
        # Write various parameters:
        (w['KohnShamStencil'],
         w['PoissonStencil'],
         w['InterpolationStencil']) = p['stencils']
        w['XCFunctional'] = paw.hamiltonian.xc.xcfunc.get_name()
        w['Charge'] = p['charge']
        w['FixMagneticMoment'] = paw.fixmom
        w['UseSymmetry'] = p['usesymm']
        w['ConvergeEmptyStates'] = p['convergeall']
        w['Converged'] = paw.converged
        w['FermiWidth'] = paw.occupation.kT
        w['MixBeta'] = paw.density.mixer.beta
        w['MixOld'] = paw.density.mixer.nmaxold
        w['MixMetric'] = paw.density.mixer.x
        w['MaximumAngularMomentum'] = paw.nuclei[0].setup.lmax
        w['SoftGauss'] = paw.nuclei[0].setup.softgauss
        w['FixDensity'] = paw.fixdensity
        w['Tolerance'] = paw.tolerance
        w['Ekin'] = paw.Ekin
        w['Epot'] = paw.Epot
        w['Ebar'] = paw.Ebar
        w['Eext'] = paw.Eext        
        w['Exc'] = paw.Exc
        w['S'] = paw.S
        epsF = paw.occupation.get_fermi_level()
        if epsF is None:
            epsF = 100.0
        w['FermiLevel'] = epsF

        # Write fingerprint (md5-digest) for all setups:
        for setup in paw.setups:
            key = names[setup.Z] + 'Fingerprint'
            if setup.type != 'paw':
                key += '(%s)' % setup.type
            w[key] = setup.fingerprint

        setup_types = p['setups']
        if isinstance(setup_types, str):
            setup_types = {None: setup_types}
        w['SetupTypes'] = repr(setup_types)
              
        typecode = {num.Float: float, num.Complex: complex}[paw.typecode]
        # write projections
        w.add('Projections', ('nspins', 'nibzkpts', 'nbands', 'nproj'),
              typecode=typecode)

        all_P_uni = num.empty((paw.nmyu, paw.nbands, nproj), paw.typecode)
        for kpt_rank in range(paw.kpt_comm.size):
            i = 0
            for nucleus in paw.nuclei:
                ni = nucleus.get_number_of_partial_waves()
                if kpt_rank == MASTER and nucleus.in_this_domain:
                    P_uni = nucleus.P_uni
                else:
                    P_uni = num.empty((paw.nmyu, paw.nbands, ni), paw.typecode)
                    world_rank = nucleus.rank + kpt_rank * paw.domain.comm.size
                    mpi.world.receive(P_uni, world_rank, 300)

                all_P_uni[:, :, i:i + ni] = P_uni
                i += ni

            for u in range(paw.nmyu):
                w.fill(all_P_uni[u])
        assert i == nproj
    else:
        for nucleus in paw.my_nuclei:
            mpi.world.send(nucleus.P_uni, MASTER, 300)

    # Write atomic density matrices and non-local part of hamiltonian:
    if mpi.rank == MASTER:
        all_D_sp = num.empty((paw.nspins, nadm), num.Float)
        all_H_sp = num.empty((paw.nspins, nadm), num.Float)
        q1 = 0
        for nucleus in paw.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            np = ni * (ni + 1) / 2
            if nucleus.in_this_domain:
                D_sp = nucleus.D_sp
                H_sp = nucleus.H_sp
            else:
                D_sp = num.empty((paw.nspins, np), num.Float)
                paw.domain.comm.receive(D_sp, nucleus.rank, 207)
                H_sp = num.empty((paw.nspins, np), num.Float)
                paw.domain.comm.receive(H_sp, nucleus.rank, 2071)
            q2 = q1 + np
            all_D_sp[:, q1:q1+np] = D_sp
            all_H_sp[:, q1:q1+np] = H_sp
            q1 = q2
        assert q2 == nadm
        w.add('AtomicDensityMatrices', ('nspins', 'nadm'), all_D_sp)
        w.add('NonLocalPartOfHamiltonian', ('nspins', 'nadm'), all_H_sp)
        
    elif paw.kpt_comm.rank == MASTER:
        for nucleus in paw.my_nuclei:
            paw.domain.comm.send(nucleus.D_sp, MASTER, 207)
            paw.domain.comm.send(nucleus.H_sp, MASTER, 2071)

    # Write the eigenvalues:
    if mpi.rank == MASTER:
        w.add('Eigenvalues', ('nspins', 'nibzkpts', 'nbands'), typecode=float)
        for kpt_rank in range(paw.kpt_comm.size):
            for u in range(paw.nmyu):
                s, k = divmod(u + kpt_rank * paw.nmyu, paw.nkpts)
                if kpt_rank == MASTER:
                    eps_n = paw.kpt_u[u].eps_n
                else:
                    eps_n = num.empty(paw.nbands, num.Float)
                    paw.kpt_comm.receive(eps_n, kpt_rank, 4300)
                w.fill(eps_n)
    elif paw.domain.comm.rank == MASTER:
        for kpt in paw.kpt_u:
            paw.kpt_comm.send(kpt.eps_n, MASTER, 4300)

    # Write the occupation numbers:
    if mpi.rank == MASTER:
        w.add('OccupationNumbers', ('nspins', 'nibzkpts', 'nbands'),
              typecode=float)
        for kpt_rank in range(paw.kpt_comm.size):
            for u in range(paw.nmyu):
                s, k = divmod(u + kpt_rank * paw.nmyu, paw.nkpts)
                if kpt_rank == MASTER:
                    f_n = paw.kpt_u[u].f_n
                else:
                    f_n = num.empty(paw.nbands, num.Float)
                    paw.kpt_comm.receive(f_n, kpt_rank, 4300)
                w.fill(f_n)
    elif paw.domain.comm.rank == MASTER:
        for kpt in paw.kpt_u:
            paw.kpt_comm.send(kpt.f_n, MASTER, 4300)

    # Write the pseudodensity on the coarse grid:
    if mpi.rank == MASTER:
        w.add('PseudoElectronDensity',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), typecode=float)
    if paw.kpt_comm.rank == MASTER:
        for s in range(paw.nspins):
            nt_sG = paw.gd.collect(paw.density.nt_sG[s])
            if mpi.rank == MASTER:
                w.fill(nt_sG)

    # Write the pseudpotential on the coarse grid:
    if mpi.rank == MASTER:
        w.add('PseudoPotential',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), typecode=float)
    if paw.kpt_comm.rank == MASTER:
        for s in range(paw.nspins):
            vt_sG = paw.gd.collect(paw.hamiltonian.vt_sG[s])
            if mpi.rank == MASTER:
                w.fill(vt_sG)

    if mode == 'all':
        # Write the wave functions:
        if mpi.rank == MASTER:
            w.add('PseudoWaveFunctions', ('nspins', 'nibzkpts', 'nbands',
                                          'ngptsx', 'ngptsy', 'ngptsz'),
                  typecode=typecode)

        for s in range(paw.nspins):
            for k in range(paw.nkpts):
                for n in range(paw.nbands):
                    psit_G = paw.get_wave_function_array(n, k, s)
                    if mpi.rank == MASTER: 
                        w.fill(psit_G)
                    
    if mpi.rank == MASTER:
        # Close the file here to ensure that the last wave function is
        # written to disk:
        w.close()

    # We don't want the slaves to start reading before the master has
    # finished writing:
    mpi.world.barrier()


def read(paw, reader):
    r = reader

    version = r['version']
    assert version >= 0.3
    
    for setup in paw.setups:
        try:
            key = names[setup.Z] + 'Fingerprint'
            if setup.type != 'paw':
                key += '(%s)' % setup.type
            fp = r[key]
        except (AttributeError, KeyError):
            break
        if setup.fingerprint != fp:
            paw.warn('Setup for %s (%s) not compatible with restart file.' %
                     (setup.symbol, setup.filename))
            
    # Read pseudoelectron density pseudo potential on the coarse grid
    # and distribute out to nodes:
    for s in range(paw.nspins): 
        paw.gd.distribute(r.get('PseudoElectronDensity', s),
                          paw.density.nt_sG[s])
    for s in range(paw.nspins): 
        paw.gd.distribute(r.get('PseudoPotential', s),
                          paw.hamiltonian.vt_sG[s])

    # Transfer the density to the fine grid:
    paw.density.interpolate_pseudo_density()

    # Read atomic density matrices and non-local part of hamiltonian:
    D_sp = r.get('AtomicDensityMatrices')
    H_sp = r.get('NonLocalPartOfHamiltonian')
    p1 = 0
    for nucleus in paw.nuclei:
        ni = nucleus.get_number_of_partial_waves()
        p2 = p1 + ni * (ni + 1) / 2
        if nucleus.in_this_domain:
            nucleus.D_sp[:] = D_sp[:, p1:p2]
            nucleus.H_sp[:] = H_sp[:, p1:p2]
        p1 = p2

    paw.Ekin = r['Ekin']
    try:
        paw.Ekin0 = r['Ekin0']
    except (AttributeError, KeyError):
        paw.Ekin0 = 0.0
    paw.Epot = r['Epot']
    paw.Ebar = r['Ebar']
    try:
        paw.Eext = r['Eext']
    except (AttributeError, KeyError):
        paw.Eext = 0.0        
    paw.Exc = r['Exc']
    paw.S = r['S']
    paw.Etot = r.get('PotentialEnergy') - 0.5 * paw.S

    paw.occupation.set_fermi_level(r['FermiLevel'])


    # Wave functions and eigenvalues:
    nkpts = len(r.get('IBZKPoints'))
    nbands = len(r.get('Eigenvalues', 0, 0))

    if nkpts == paw.nkpts:
        for kpt in paw.kpt_u:
            kpt.allocate(nbands)
            # Eigenvalues and occupation numbers:
            k = kpt.k
            s = kpt.s
            kpt.eps_n[:] = r.get('Eigenvalues', s, k)
            kpt.f_n[:] = r.get('OccupationNumbers', s, k)
        
        if r.has_array('PseudoWaveFunctions'):
            if mpi.parallel:
                wait
                # Slice of the global array for this domain:
                i = [slice(b - 1 + p, e - 1 + p) for b, e, p in
                     zip(paw.gd.beg_c, paw.gd.end_c, paw.gd.domain.periodic_c)]

                for kpt in paw.kpt_u:
                    kpt.psit_nG = paw.gd.empty(nbands, paw.typecode)
                    # Read band by band to save memory
                    for n, psit_G in enumerate(kpt.psit_nG):
                        psit_G[:] = r.get('PseudoWaveFunctions',
                                          kpt.s, kpt.k, n)[i]
            else:
                # Serial calculation.  We may not be able to keep all the wave
                # functions in memory - so psit_nG will be a special type of
                # array that is really just a reference to a file:
                for kpt in paw.kpt_u:
                    kpt.psit_nG = r.get_reference('PseudoWaveFunctions',
                                                  kpt.s, kpt.k)
    
        for u, kpt in enumerate(paw.kpt_u):
            P_ni = r.get('Projections', kpt.s, kpt.k)
            i1 = 0
            for nucleus in paw.nuclei:
                i2 = i1 + nucleus.get_number_of_partial_waves()
                if nucleus.in_this_domain:
                    nucleus.P_uni[u, :nbands] = P_ni[:, i1:i2]
                i1 = i2

    # Get the forces from the old calculation:
    if r.has_array('CartesianForces'):
        paw.F_ac = r.get('CartesianForces')

    #r.close()
