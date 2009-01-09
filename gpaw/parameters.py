from ase.units import Hartree

from gpaw.poisson import PoissonSolver


class InputParameters(dict):
    def __init__(self):
        dict.__init__(self, [
            ('h',             None,   ),  # Angstrom
            ('xc',            'LDA',  ),
            ('gpts',          None,   ),
            ('kpts',          None,   ),
            ('lmax',          2,      ),
            ('charge',        0,      ),
            ('fixmom',        False,  ),
            ('nbands',        None,   ),
            ('setups',        'paw',  ),
            ('basis',         {},     ),
            ('width',         None,   ),  # eV
            ('spinpol',       None,   ),
            ('usesymm',       True,   ),
            ('stencils',      (2, 3), ),
            ('fixdensity',    False   ),
            ('mixer',         None,   ),
            ('txt',           '-',    ),
            ('hund',          False,  ),
            ('random',        False,  ),
            ('maxiter',       120,    ),
            ('parsize',       None,   ),
            ('parsize_bands', 1   ,   ),
            ('external',      None,   ),  # eV
            ('verbose',       0,      ),
            ('eigensolver',   None,   ),
            ('poissonsolver', None,   ),
            ('communicator' , None,   ),
            ('idiotproof'   , True,   ),
            ('mode',          'fd',   ),
            ('convergence',   {'energy':      0.001,  # eV / atom
                               'density':     1.0e-3,
                               'eigenstates': 1.0e-9,
                               'bands':       'occupied'})])
    
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def read(self, reader):
        """Read state from file."""

        if isinstance(reader, str):
            reader = gpaw.io.open(reader, 'r')

        r = reader

        version = r['version']
        
        assert version >= 0.3
    
        self.xc = r['XCFunctional']
        self.nbands = r.dimension('nbands')
        self.spinpol = (r.dimension('nspins') == 2)
        self.kpts = r.get('BZKPoints')
        self.usesymm = r['UseSymmetry']
        self.gpts = ((r.dimension('ngptsx') + 1) // 2 * 2,
                     (r.dimension('ngptsy') + 1) // 2 * 2,
                     (r.dimension('ngptsz') + 1) // 2 * 2)
        self.lmax = r['MaximumAngularMomentum']
        self.setups = r['SetupTypes']
        self.fixdensity = r['FixDensity']
        if version <= 0.4:
            # Old version: XXX
            print('# Warning: Reading old version 0.3/0.4 restart files ' +
                  'will be disabled some day in the future!')
            self.convergence['eigenstates'] = r['Tolerance']
        else:
            self.convergence = {'density': r['DensityConvergenceCriterion'],
                                'energy':
                                r['EnergyConvergenceCriterion'] * Hartree,
                                'eigenstates':
                                r['EigenstatesConvergenceCriterion'],
                                'bands': r['NumberOfBandsToConverge']}
            if version <= 0.6:
                mixer = 'Mixer'
                weight = r['MixMetric']
                if weight == 1.0:
                    metric = None
                else:
                    metric = 'old'
            else:
                mixer = r['MixClass']
                weight = r['MixWeight']
                metric = r['MixMetric']

            if mixer == 'Mixer':
                from gpaw.mixer import Mixer
            elif mixer == 'MixerSum':
                from gpaw.mixer import MixerSum as Mixer
            else:
                Mixer = None

            if Mixer is None:
                self.mixer = None
            else:
                self.mixer = Mixer(r['MixBeta'], r['MixOld'], metric, weight)
            
        if version == 0.3:
            # Old version: XXX
            print('# Warning: Reading old version 0.3 restart files is ' +
                  'dangerous and will be disabled some day in the future!')
            self.stencils = (2, 3)
            self.charge = 0.0
            self.fixmom = False
        else:
            self.stencils = (r['KohnShamStencil'],
                             r['InterpolationStencil'])
            self.poissonsolver = PoissonSolver(nn=r['PoissonStencil'])
            self.charge = r['Charge']
            self.fixmom = r['FixMagneticMoment']

        self.width = r['FermiWidth'] * Hartree

        self.mode = r.parameters.get('Mode', 'fd')
