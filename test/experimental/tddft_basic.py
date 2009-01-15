
import unittest
import os.path
from numpy import ndarray,zeros,any,all,abs

from ase import Atoms,Atom
from gpaw import Calculator

# -------------------------------------------------------------------

class GammaPointSetup(unittest.TestCase):

    def setUp(self):
        self.restartfile = 'ethyne_ground.gpw'
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
        # Bond lengths between H-C and C-C
        dhc = 1.06
        dcc = 0.6612

        self.atoms = Atoms([Atom('H', (0, 0, 0)),
                    Atom('C', (dhc, 0, 0)),
                    Atom('C', (dhc+dcc, 0, 0)),
                    Atom('H', (2*dhc+dcc, 0, 0))])

        self.atoms.center(vacuum=2.0)

        self.calc = Calculator(nbands=int(10/2.0)+4,
                    h = 0.1,
                    txt='ethyne_ground.txt')

        self.atoms.set_calculator(self.calc)

    # =================================

    def test_consistency(self):

        self.assertEqual(self.calc.initialized,self.restarted)
        self.assertEqual(self.calc.converged,self.restarted)

        Epot = self.atoms.get_potential_energy()

        self.assertAlmostEqual(Epot,27.5186,places=4)

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.converged)
                
# -------------------------------------------------------------------

from gpaw.tddft import TDDFT

class GammaPointTDDFT(GammaPointSetup):

    tolerance = 1e-8

    def setUp(self):
        GammaPointSetup.setUp(self)
        self.assertTrue(self.restarted)

        self.tdcalc = TDDFT(self.restartfile, txt='ethyne_tddft.txt',
                    propagator=self.propagator, solver=self.solver,
                    tolerance=self.tolerance)

        self.time_step = 10.0    # 1 attoseconds = 0.041341 autime
        self.iterations = 3      # 3 x 10 as => 1.24023 autime

    def tearDown(self):
        GammaPointSetup.tearDown(self)
        del self.tdcalc

    # =================================

    def test_propagation(self):
        # Propagate without saving the time-dependent dipole moment
        # to a .dat-file, nor periodically dumping a restart file
        self.tdcalc.propagate(self.time_step, self.iterations)

# -------------------------------------------------------------------

class GammaPointTDDFT_ECN_CSCG(GammaPointTDDFT):
    propagator = 'ECN'
    solver = 'CSCG'

class GammaPointTDDFT_SICN_CSCG(GammaPointTDDFT):
    propagator = 'SICN'
    solver = 'CSCG'

class GammaPointTDDFT_SITE_CSCG(GammaPointTDDFT):
    propagator = 'SITE'
    solver = 'CSCG'

class GammaPointTDDFT_SIKE_CSCG(GammaPointTDDFT):
    propagator = 'SIKE'
    solver = 'CSCG'

# -------------------------------------------------------------------

if __name__ == '__main__':
    #unittest.main()

    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [GammaPointTDDFT_ECN_CSCG, GammaPointTDDFT_SICN_CSCG,
                GammaPointTDDFT_SITE_CSCG, GammaPointTDDFT_SIKE_CSCG]

    for test in testcases:
        testrunner.run(unittest.defaultTestLoader.loadTestsFromTestCase(test))


