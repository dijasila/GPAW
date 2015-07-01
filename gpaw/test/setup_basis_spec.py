import numpy as np
from ase import Atoms
from gpaw import GPAW, setup_paths
from gpaw.setup_data import SetupData

# This test looks for differently named setups and basis sets to ascertain
# that correct filenames are constructed.  Generally speaking, the things
# it looks for do not exist; we just verify the filenames.
del setup_paths[:]

system = Atoms('Na')
system.center(vacuum=3.0)

def check(setups, basis, refname):
    calc = GPAW(setups=setups, basis=basis)
    system.set_calculator(calc)
    try:
        calc.initialize(system)
    except RuntimeError as err:
        msg = str(err)
        fname = msg.split('"')[1]
        assert fname == refname, fname
    else:
        assert False

check({}, 'hello', 'Na.hello.basis')
check('hello', {}, 'Na.hello.LDA')
check('hello', 'dzp', 'Na.hello.dzp.basis')
check('hello', 'sz(dzp)', 'Na.hello.dzp.basis')
check('hello', 'world.dzp', 'Na.hello.world.dzp.basis')
check('hello', 'sz(world.dzp)', 'Na.hello.world.dzp.basis')
check('paw', 'world.dzp', 'Na.world.dzp.basis')
check('paw', 'sz(world.dzp)', 'Na.world.dzp.basis')
