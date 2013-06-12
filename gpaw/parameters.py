import numpy as np

from ase.units import Hartree, Bohr
from ase.dft.kpoints import monkhorst_pack

from gpaw.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.occupations import FermiDirac
from gpaw.wavefunctions.pw import PW


def read_parameters(params, reader):
    """Read state from file."""

    r = reader

    version = r['version']
    
    assert version >= 0.3

    params.xc = r['XCFunctional']
    params.nbands = r.dimension('nbands')
    params.spinpol = (r.dimension('nspins') == 2)

    bzk_kc = r.get('BZKPoints', broadcast=True)
    if r.has_array('NBZKPoints'):
        params.kpts = r.get('NBZKPoints', broadcast=True)
        if r.has_array('MonkhorstPackOffset'):
            offset_c = r.get('MonkhorstPackOffset', broadcast=True)
            if offset_c.any():
                params.kpts = monkhorst_pack(params.kpts) + offset_c
    else:
        params.kpts = bzk_kc

    params.usesymm = r['UseSymmetry']
    params.basis = r['BasisSet']

    if version >= 4:
        params.realspace = r['RealSpace']

    if version >= 2:
        try:
            h = r['GridSpacing']
        except KeyError:  # CMR can't handle None!
            h = None
        if h is not None:
            params.h = Bohr * h
        if r.has_array('GridPoints'):
            params.gpts = r.get('GridPoints')
    else:
        if version >= 0.9:
            h = r['GridSpacing']
        else:
            h = None

        gpts = ((r.dimension('ngptsx') + 1) // 2 * 2,
                (r.dimension('ngptsy') + 1) // 2 * 2,
                (r.dimension('ngptsz') + 1) // 2 * 2)

        if h is None:
            params.gpts = gpts
        else:
            params.h = Bohr * h

    params.lmax = r['MaximumAngularMomentum']
    params.setups = r['SetupTypes']
    assert params.setups is not None
    params.fixdensity = r['FixDensity']
    if version <= 0.4:
        # Old version: XXX
        print('# Warning: Reading old version 0.3/0.4 restart files ' +
              'will be disabled some day in the future!')
        params.convergence['eigenstates'] = r['Tolerance']
    else:
        nbtc = r['NumberOfBandsToConverge']
        if not isinstance(nbtc, (int, str)):
            # The string 'all' was eval'ed to the all() function!
            nbtc = 'all'
        params.convergence = {'density': r['DensityConvergenceCriterion'],
                            'energy':
                            r['EnergyConvergenceCriterion'] * Hartree,
                            'eigenstates':
                            r['EigenstatesConvergenceCriterion'],
                            'bands': nbtc}

        if version < 1:
            # Volume per grid-point:
            dv = (abs(np.linalg.det(r.get('UnitCell'))) /
                  (gpts[0] * gpts[1] * gpts[2]))
            params.convergence['eigenstates'] *= Hartree**2 * dv

        if version <= 0.6:
            mixer = 'Mixer'
            weight = r['MixMetric']
        elif version <= 0.7:
            mixer = r['MixClass']
            weight = r['MixWeight']
            metric = r['MixMetric']
            if metric is None:
                weight = 1.0
        else:
            mixer = r['MixClass']
            weight = r['MixWeight']

        if mixer == 'Mixer':
            from gpaw.mixer import Mixer
        elif mixer == 'MixerSum':
            from gpaw.mixer import MixerSum as Mixer
        elif mixer == 'MixerSum2':
            from gpaw.mixer import MixerSum2 as Mixer
        elif mixer == 'MixerDif':
            from gpaw.mixer import MixerDif as Mixer
        elif mixer == 'DummyMixer':
            from gpaw.mixer import DummyMixer as Mixer
        else:
            Mixer = None

        if Mixer is None:
            params.mixer = None
        else:
            params.mixer = Mixer(r['MixBeta'], r['MixOld'], weight)
        
    if version == 0.3:
        # Old version: XXX
        print('# Warning: Reading old version 0.3 restart files is ' +
              'dangerous and will be disabled some day in the future!')
        params.stencils = (2, 3)
        params.charge = 0.0
        fixmom = False
    else:
        params.stencils = (r['KohnShamStencil'],
                         r['InterpolationStencil'])
        if r['PoissonStencil'] == 999:
            params.poissonsolver = FFTPoissonSolver()
        else:
            params.poissonsolver = PoissonSolver(nn=r['PoissonStencil'])
        params.charge = r['Charge']
        fixmom = r['FixMagneticMoment']

    if version < 4:
        params.occupations = FermiDirac(r['FermiWidth'] * Hartree,
                                        fixmagmom=fixmom)
    else:
        params.smearing = {'type': r['SmearingType'],
                           'width': r['SmearingWidth'] * Hartree}
        if fixmom:
            params.smearing['fixmagmom'] = True

    try:
        params.mode = r['Mode']
    except KeyError:
        params.mode = 'fd'

    if params.mode == 'pw':
        params.mode = PW(ecut=r['PlaneWaveCutoff'] * Hartree)
        
    if len(bzk_kc) == 1 and not bzk_kc[0].any():
        # Gamma point only:
        if r['DataType'] == 'Complex':
            params.dtype = complex
