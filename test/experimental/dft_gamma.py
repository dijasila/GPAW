
import unittest
import os.path

from ase import Atoms,Atom
from gpaw import Calculator

#from numpy import ndarray,zeros,any,all,abs

# -------------------------------------------------------------------

class UTGammaPointSetup(unittest.TestCase):
    """
    Setup a simple gamma point calculation with DFT."""

    def setUp(self):
        self.restartfile = 'dft_gamma.gpw'
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
                    txt='dft_gamma.txt')

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

if __name__ == '__main__':
    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTGammaPointSetup]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)

