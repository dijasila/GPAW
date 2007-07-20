"""
A simple test script.

Runs all scripts named ``*.py``.  The tests that execute
the fastest will be run first.
"""

import os
import sys
import unittest
from glob import glob
import gc
from optparse import OptionParser


parser = OptionParser(usage='%prog [options] [tests]',
                      version='%prog 0.1')

parser.add_option('-v', '--verbosity',
                  type='int', default=2,
                  help='Verbocity level.')

parser.add_option('-x', '--exclude',
                  type='string', default=None,
                  help='Exclude tests (comma separated list of tests).',
                  metavar='test1.py,test2.py,...')

parser.add_option('-f', '--run-failed-tests-only',
                  action='store_true',
                  help='Run failed tests only.')

parser.add_option('-d', '--debug',
                  action='store_true', default=False,
                  help='Run tests in debug mode.')

parser.add_option('-p', '--parallel',
                  action='store_true',
                  help='Add parallel tests.')

opt, tests = parser.parse_args()

if len(tests) == 0:
    tests = ['setups.py',  'pbe-pw91.py',  'xcfunc.py',  'gradient.py',
             'xc.py',  'gp2.py',  'Gauss.py',  'non-periodic.py',  'lf.py',
             'denom_int.py',  'transformations.py',  'XC2.py',  'poisson.py',
             'XC2Spin.py',  'integral4.py',  'd2Excdn2.py',
             'multipoletest.py',  'proton.py',  'restart.py',  'timing.py',
             'xcatom.py',  'coulomb.py',  'nonselfconsistentLDA.py',
             'kli.py',  'units.py',  'revPBE.py',  'nonselfconsistent.py',
             'hydrogen.py',  'spinpol.py',  'stdout.py',  'gga-atom.py',
             'gauss_func.py',  'H-force.py',  'degeneracy.py',  'cg.py',
             'h2o-xas.py',  'davidson.py',  'wannier-ethylene.py',
             'restart2.py',  'CH4.py',  'gllb2.py',  'lrtddft.py',
             'fixmom.py',  'wannier-hwire.py',  'exx.py',  'revPBE_Li.py',
             'td_hydrogen.py',  'aedensity.py',  'IP-oxygen.py',  '2Al.py',
             '8Si.py',  'Cu.py',  'ltt.py',  'generatesetups.py',
             'ae-calculation.py',  'H2Al110.py']
    tests_lxc = [
        'lxc_spinpol_Li.py', 'lxc_xcatom.py'
        ]
    tests = tests + tests_lxc


if opt.run_failed_tests_only:
    tests = [line.strip() for line in open('failed-tests.txt')]

if opt.debug:
    sys.argv.append('--gpaw-debug')

exclude = ['__init__.py', 'test.py', 'C-force.py', 'grr.py']
if opt.exclude is not None:
    exclude += opt.exclude.split(',')

# exclude parallel tests if opt.parallel is not set
if not opt.parallel:
    exclude.extend(['parallel-restart.py', 'parmigrate.py',
                    'par8.py', 'par6.py', 'exx_parallel.py'])

for test in exclude:
    if test in tests:
        tests.remove(test)

#gc.set_debug(gc.DEBUG_SAVEALL)

from ASE.Units import units

class ScriptTestCase(unittest.TestCase):
    garbage = []
    def __init__(self, filename):
        unittest.TestCase.__init__(self, 'testfile')
        self.filename = filename

    def setUp(self):
        units.length_used = False
        units.energy_used = False
        units.SetUnits('Ang', 'eV')

    def testfile(self):
        execfile(self.filename, {})

    def tearDown(self):
        gc.collect()
        n = len(gc.garbage)
        ScriptTestCase.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s)' %
                        (n, 's'[:n > 1]))

    def id(self):
        return self.filename

    def __str__(self):
        return '%s' % self.filename

    def __repr__(self):
        return "ScriptTestCase('%s')" % self.filename


ts = unittest.TestSuite()
for test in tests:
    ts.addTest(ScriptTestCase(filename=test))

from gpaw.utilities import DownTheDrain
sys.stdout = DownTheDrain()

ttr = unittest.TextTestRunner(verbosity=opt.verbosity)
result = ttr.run(ts)
failed = [test.filename for test, msg in result.failures + result.errors]

sys.stdout = sys.__stdout__

if len(failed) > 0:
    open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
