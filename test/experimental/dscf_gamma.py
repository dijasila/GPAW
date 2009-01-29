
import unittest
import os.path

from ase import Atoms,Atom
from gpaw import Calculator
from gpaw.mixer import Mixer,MixerSum

#from numpy import ndarray,zeros,any,all,abs

# -------------------------------------------------------------------

class UTGammaPointSetup_DSCFGroundState(unittest.TestCase):
    """
    Setup a DSCF-compatible ground state gamma point calculation with DFT."""

    def setUp(self):
        self.restartfile = 'dscf_gamma_ground.gpw'
        self.restarted = os.path.isfile(self.restartfile)

        if self.restarted:
            self.calc = Calculator(self.restartfile,txt=None)
            self.atoms = self.calc.get_atoms()
        else:
            self.initialize()

    def tearDown(self):
        if not self.restarted:
            self.calc.write(self.restartfile, mode='all')

        del self.atoms
        del self.calc

    # =================================

    def initialize(self):
        # Bond lengths between H-C and C-C for ethyne (acetylene) cf.
        # CRC Handbook of Chemistry and Physics, 87th ed., p. 9-28
        dhc = 1.060
        dcc = 1.203

        self.atoms = Atoms([Atom('H', (0, 0, 0)),
                    Atom('C', (dhc, 0, 0)),
                    Atom('C', (dhc+dcc, 0, 0)),
                    Atom('H', (2*dhc+dcc, 0, 0))])

        self.atoms.center(vacuum=4.0)

        # Number of occupied and unoccupied bands to converge
        nbands = int(10/2.0)+3

        # Number of additional bands for DSCF linear expansion
        nextra = 10

        self.calc = Calculator(h=0.2,
                    nbands=nbands+nextra,
                    xc='RPBE',
                    spinpol=True,
                    eigensolver='cg',
                    mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
                    convergence={'eigenstates': 1e-9, 'bands':nbands},
                    width=0.1, #TODO might help convergence?
                    txt='dscf_gamma_ground.txt')

        self.atoms.set_calculator(self.calc)

    # =================================

    def test_consistency(self):

        self.assertEqual(self.calc.initialized,self.restarted)
        self.assertEqual(self.calc.converged,self.restarted)

        Epot = self.atoms.get_potential_energy()

        self.assertAlmostEqual(Epot,-22.8126,places=4)

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.converged)

    def test_degeneracy(self):

        degeneracies = [(3,4),(5,6)]

        for kpt in self.calc.kpt_u:
            for (a,b) in degeneracies:
                self.assertAlmostEqual(kpt.eps_n[a],kpt.eps_n[b],places=6)

    def test_occupancy(self):

        ne_u = [5., 5.]

        for kpt in self.calc.kpt_u:
            self.assertAlmostEqual(sum(kpt.f_n),ne_u[kpt.u],places=4)

# -------------------------------------------------------------------

class UTGammaPointSetup_DSCFExcitedState(unittest.TestCase):
    """
    Setup an excited state gamma point calculation with DSCF."""

    def setUp(self):
        self.restartfile = 'dscf_gamma_excited.gpw'
        self.restarted = os.path.isfile(self.restartfile)

        if self.restarted:
            self.calc = Calculator(self.restartfile,txt=None)
            self.atoms = self.calc.get_atoms()
        else:
            self.initialize()

    def tearDown(self):
        if not self.restarted:
            self.calc.write(self.restartfile, mode='all')

        del self.atoms
        del self.calc

    # =================================

    def initialize(self):
        # Construct would-be wavefunction from ground state gas phase calculation
        self.calc = Calculator('dscf_gamma_ground.gpw',
                    #eigensolver='cg',
                    #mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
                    #mixer=MixerSum(beta=0.1, nmaxold=5, metric='new', weight=100),
                    #mixer=Mixer(beta=0.1, nmaxold=5, metric='new', weight=100),
                    #convergence={'eigenstates': 1e-7, 'bands':nbands},
                    txt='dscf_gamma_excited.txt')

        self.atoms = self.calc.get_atoms()

        from gpaw.utilities.dscftools import dscf_find_atoms,dscf_linear_combination

        mol = dscf_find_atoms(self.atoms,'C')
        sel_n = [5,6] #TODO!!!
        coeff_n = [1/2**0.5,1j/2**0.5]
        #sel_n = [5]
        #coeff_n = [1.] #TODO!!!

        (P_aui,wf_u,) = dscf_linear_combination(self.calc,mol,sel_n,coeff_n)

        from gpaw.dscf import WaveFunction,dscf_calculation

        # Setup dSCF calculation to occupy would-be wavefunction
        sigma_star = WaveFunction(self.calc,wf_u,P_aui,molecule=mol)

        # Force one electron (spin down) into the sigma star orbital
        dscf_calculation(self.calc, [[1.0,sigma_star,1]], self.atoms)

        # TODO DEBUG TEST
        self.calc.print_parameters()

    # =================================

    def test_consistency(self):

        self.assertTrue(self.calc.initialized)
        self.assertEqual(self.calc.converged,self.restarted)

        Epot = self.atoms.get_potential_energy()

        self.assertAlmostEqual(Epot,-17.1042,places=4)

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.converged)

    def test_degeneracy(self):

        #TODO apparently the px/py-degeneracy is lifted for both spins?
        degeneracies = []

        for kpt in self.calc.kpt_u:
            for (a,b) in degeneracies:
                self.assertAlmostEqual(kpt.eps_n[a],kpt.eps_n[b],places=6)

    def test_occupancy(self):

        ne_u = [4., 5.]

        for kpt in self.calc.kpt_u:
            self.assertAlmostEqual(sum(kpt.f_n),ne_u[kpt.u],places=4)

# -------------------------------------------------------------------

if __name__ == '__main__':
    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTGammaPointSetup_DSCFGroundState,UTGammaPointSetup_DSCFExcitedState]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)

