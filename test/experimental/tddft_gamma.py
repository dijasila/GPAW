
import unittest

from gpaw.output import Output
from gpaw.tddft import TDDFT

from dft_gamma import UTGammaPointSetup

# -------------------------------------------------------------------

debug = Output()
debug.set_text('tddft_gamma.log')

class UTGammaPointTDDFT(UTGammaPointSetup):
    """
    Propagate a gamma point calculation with TDDFT."""

    tolerance = 1e-8

    def setUp(self):
        UTGammaPointSetup.setUp(self)
        self.assertTrue(self.restarted)

        self.tdcalc = TDDFT(self.restartfile, txt='tddft_gamma.txt',
                    propagator=self.propagator, solver=self.solver,
                    tolerance=self.tolerance, debug=debug)

        self.time_step = 10.0    # 1 attoseconds = 0.041341 autime
        self.iterations = 3      # 3 x 10 as => 1.24023 autime

    def tearDown(self):
        del self.tdcalc
        UTGammaPointSetup.tearDown(self)

    # =================================

    def test_propagation(self):
        # Propagate without saving the time-dependent dipole moment
        # to a .dat-file, nor periodically dumping a restart file
        self.tdcalc.propagate(self.time_step, self.iterations)

# -------------------------------------------------------------------

class UTGammaPointTDDFT_ECN_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is ECN and solver CSCG."""

    propagator = 'ECN'
    solver = 'CSCG'

class UTGammaPointTDDFT_SICN_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is SICN and solver CSCG."""
    propagator = 'SICN'
    solver = 'CSCG'

class UTGammaPointTDDFT_SITE_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is SITE and solver CSCG."""
    propagator = 'SITE'
    solver = 'CSCG'

class UTGammaPointTDDFT_SIKE_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is SIKE6 and solver CSCG."""
    propagator = 'SIKE6'
    solver = 'CSCG'

# -------------------------------------------------------------------

if __name__ == '__main__':

    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTGammaPointTDDFT_ECN_CSCG, UTGammaPointTDDFT_SICN_CSCG,
                UTGammaPointTDDFT_SITE_CSCG, UTGammaPointTDDFT_SIKE_CSCG]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)


