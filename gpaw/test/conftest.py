import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from _pytest.tmpdir import _mk_tmp
from ase import Atoms
from ase.build import bulk
from ase.lattice.hexagonal import Graphene
from ase.io import read
from gpaw import GPAW, PW, Davidson, FermiDirac, setup_paths
from gpaw.cli.info import info
from gpaw.mpi import broadcast, world
from gpaw.utilities import devnull


@contextmanager
def execute_in_tmp_path(request, tmp_path_factory):
    if world.rank == 0:
        # Obtain basename as
        # * request.function.__name__  for function fixture
        # * request.module.__name__    for module fixture
        basename = getattr(request, request.scope).__name__
        path = tmp_path_factory.mktemp(basename)
    else:
        path = None
    path = broadcast(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='function')
def in_tmp_dir(request, tmp_path_factory):
    """Run test function in a temporary directory."""
    with execute_in_tmp_path(request, tmp_path_factory) as path:
        yield path


@pytest.fixture(scope='module')
def module_tmp_path(request, tmp_path_factory):
    """Run test module in a temporary directory."""
    with execute_in_tmp_path(request, tmp_path_factory) as path:
        yield path


@pytest.fixture
def add_cwd_to_setup_paths():
    """Temporarily add current working directory to setup_paths."""
    try:
        setup_paths[:0] = ['.']
        yield
    finally:
        del setup_paths[:1]


@pytest.fixture(scope='session')
def gpw_files(request, tmp_path_factory):
    """Reuse gpw-files.

    Returns a dict mapping names to paths to gpw-files.  If you
    want to reuse gpw-files from an earlier pytest session then set the
    ``$GPW_TEST_FILES`` environment variable and the files will be written
    to that folder.

    Example::

        def test_something(gpw_files):
            calc = GPAW(gpw_files['h2_lcao_wfs'])
            ...

    Possible systems are:

    * Bulk BCC-Li with 3x3x3 k-points: ``bcc_li_pw``, ``bcc_li_fd``,
      ``bcc_li_lcao``.

    * O2 molecule: ``o2_pw``.

    * H2 molecule: ``h2_pw``, ``h2_fd``, ``h2_lcao``.

    * H2 molecule (not centered): ``h2_pw_0``.

    * Spin-polarized H atom: ``h_pw``.

    * Polyethylene chain.  One unit, 3 k-points, no symmetry:
      ``c2h4_pw_nosym``.  Three units: ``c6h12_pw``.

    * Bulk TiO2 with 4x4x4 k-points: ``ti2o4_pw`` and ``ti2o4_pw_nosym``.

    * Bulk BN (zinkblende) with 2x2x2 k-points and 9 converged bands:
      ``bn_pw``.

    * h-BN layer with 3x3x1 (gamma center) k-points and 26 converged bands:
      ``hbn_pw``.

    * Graphene with 6x6x1 k-points: ``graphene_pw``

    * MoS2 with 6x6x1 k-points: ``mos2_pw``

    * Bulk Si, LDA, 2x2x2 k-points (gamma centered): ``si_pw``

    * Bulk Fe, LDA, 4x4x4 k-points, 6 converged bands: ``fe_pw``

    Files with wave functions are also available (add ``_wfs`` to the names).
    """
    path = os.environ.get('GPW_TEST_FILES')
    if not path:
        warnings.warn(
            'Note that you can speed up the tests by reusing gpw-files '
            'from an earlier pytest session: '
            'set the $GPW_TEST_FILES environment variable and the '
            'files will be written to/read from that folder. '
            'See: https://wiki.fysik.dtu.dk/gpaw/devel/testing.html'
            '#gpaw.test.conftest.gpw_files')
        if world.rank == 0:
            path = _mk_tmp(request, tmp_path_factory)
        else:
            path = None
        path = broadcast(path)
    return GPWFiles(Path(path))


class GPWFiles:
    """Create gpw-files."""
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(exist_ok=True)
        self.gpw_files = {}
        for file in path.glob('*.gpw'):
            self.gpw_files[file.name[:-4]] = file

    def __getitem__(self, name: str) -> Path:
        if name not in self.gpw_files:
            rawname, _, _ = name.partition('_wfs')
            calc = getattr(self, rawname)()
            path = self.path / (rawname + '.gpw')
            calc.write(path)
            self.gpw_files[rawname] = path
            path = self.path / (rawname + '_wfs.gpw')
            calc.write(path, mode='all')
            self.gpw_files[rawname + '_wfs'] = path
        return self.gpw_files[name]

    def bcc_li_pw(self):
        return self.bcc_li({'name': 'pw', 'ecut': 200})

    def bcc_li_fd(self):
        return self.bcc_li({'name': 'fd'})

    def bcc_li_lcao(self):
        return self.bcc_li({'name': 'lcao'})

    def bcc_li(self, mode):
        li = bulk('Li', 'bcc', 3.49)
        li.calc = GPAW(mode=mode,
                       kpts=(3, 3, 3),
                       txt=self.path / f'bcc_li_{mode["name"]}.txt')
        li.get_potential_energy()
        return li.calc

    def h2_pw(self):
        return self.h2({'name': 'pw', 'ecut': 200})

    def h2_fd(self):
        return self.h2({'name': 'fd'})

    def h2_lcao(self):
        return self.h2({'name': 'lcao'})

    def h2(self, mode):
        h2 = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
        h2.center(vacuum=2.5)
        h2.calc = GPAW(mode=mode,
                       txt=self.path / f'h2_{mode["name"]}.txt')
        h2.get_potential_energy()
        return h2.calc

    def h2_pw_0(self):
        h2 = Atoms('H2',
                   positions=[[-0.37, 0, 0], [0.37, 0, 0]],
                   cell=[5.74, 5, 5],
                   pbc=True)
        h2.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                       txt=self.path / 'h2_pw_0.txt')
        h2.get_potential_energy()
        return h2.calc

    def h_pw(self):
        h = Atoms('H', magmoms=[1])
        h.center(vacuum=4.0)
        h.calc = GPAW(mode={'name': 'pw', 'ecut': 500},
                      txt=self.path / 'h_pw.txt')
        h.get_potential_energy()
        return h.calc

    def o2_pw(self):
        d = 1.1
        a = Atoms('O2', positions=[[0, 0, 0], [d, 0, 0]], magmoms=[1, 1])
        a.center(vacuum=4.0)
        a.calc = GPAW(mode={'name': 'pw', 'ecut': 800},
                      txt=self.path / 'o2_pw.txt')
        a.get_potential_energy()
        return a.calc

    def co_lcao(self):
        d = 1.1
        co = Atoms('CO', positions=[[0, 0, 0], [d, 0, 0]])
        co.center(vacuum=4.0)
        co.calc = GPAW(mode='lcao',
                       txt=self.path / 'co_lcao.txt')
        co.get_potential_energy()
        return co.calc

    def c2h4_pw_nosym(self):
        d = 1.54
        h = 1.1
        x = d * (2 / 3)**0.5
        z = d / 3**0.5
        pe = Atoms('C2H4',
                   positions=[[0, 0, 0],
                              [x, 0, z],
                              [0, -h * (2 / 3)**0.5, -h / 3**0.5],
                              [0, h * (2 / 3)**0.5, -h / 3**0.5],
                              [x, -h * (2 / 3)**0.5, z + h / 3**0.5],
                              [x, h * (2 / 3)**0.5, z + h / 3**0.5]],
                   cell=[2 * x, 0, 0],
                   pbc=(1, 0, 0))
        pe.center(vacuum=2.0, axis=(1, 2))
        pe.calc = GPAW(mode='pw',
                       kpts=(3, 1, 1),
                       symmetry='off',
                       txt=self.path / 'c2h4_pw_nosym.txt')
        pe.get_potential_energy()
        return pe.calc

    def c6h12_pw(self):
        pe = read(self['c2h4_pw_nosym'])
        pe = pe.repeat((3, 1, 1))
        pe.calc = GPAW(mode='pw', txt=self.path / 'c6h12_pw.txt')
        pe.get_potential_energy()
        return pe.calc

    def h2o_lcao(self):
        from ase.build import molecule
        atoms = molecule('H2O', cell=[8, 8, 8], pbc=1)
        atoms.center()
        atoms.calc = GPAW(mode='lcao', txt=self.path / 'h2o.txt')
        atoms.get_potential_energy()
        return atoms.calc

    def ti2o4(self, symmetry):
        pwcutoff = 400.0
        k = 4
        a = 4.59
        c = 2.96
        u = 0.305

        rutile_cell = [[a, 0, 0],
                       [0, a, 0],
                       [0, 0, c]]

        TiO2_basis = np.array([[0.0, 0.0, 0.0],
                               [0.5, 0.5, 0.5],
                               [u, u, 0.0],
                               [-u, -u, 0.0],
                               [0.5 + u, 0.5 - u, 0.5],
                               [0.5 - u, 0.5 + u, 0.5]])

        bulk_crystal = Atoms(symbols='Ti2O4',
                             scaled_positions=TiO2_basis,
                             cell=rutile_cell,
                             pbc=(1, 1, 1))

        tag = '_nosym' if symmetry == 'off' else ''
        bulk_calc = GPAW(mode=PW(pwcutoff),
                         nbands=42,
                         eigensolver=Davidson(1),
                         kpts={'size': (k, k, k), 'gamma': True},
                         xc='PBE',
                         occupations=FermiDirac(0.00001),
                         parallel={'band': 1},
                         symmetry=symmetry,
                         txt=self.path / f'ti2o4_pw{tag}.txt')

        bulk_crystal.calc = bulk_calc
        bulk_crystal.get_potential_energy()
        return bulk_calc

    def ti2o4_pw(self):
        return self.ti2o4({})

    def ti2o4_pw_nosym(self):
        return self.ti2o4('off')

    def si_pw(self):
        si = bulk('Si')
        calc = GPAW(mode='pw',
                    xc='LDA',
                    occupations=FermiDirac(width=0.001),
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    txt='si.gs.txt')
        si.calc = calc
        si.get_potential_energy()
        return si.calc

    def bn_pw(self):
        atoms = bulk('BN', 'zincblende', a=3.615)
        atoms.calc = GPAW(mode=PW(400),
                          kpts={'size': (2, 2, 2), 'gamma': True},
                          nbands=12,
                          convergence={'bands': 9},
                          occupations=FermiDirac(0.001),
                          txt=self.path / 'bn_pw.txt')
        atoms.get_potential_energy()
        return atoms.calc

    def hbn_pw(self):
        atoms = Graphene(symbol='B',
                         latticeconstant={'a': 2.5, 'c': 1.0},
                         size=(1, 1, 1))
        atoms[0].symbol = 'N'
        atoms.pbc = (1, 1, 0)
        atoms.center(axis=2, vacuum=3.0)
        atoms.calc = GPAW(mode=PW(400),
                          xc='LDA',
                          nbands=50,
                          occupations=FermiDirac(0.001),
                          parallel={'domain': 1},
                          convergence={'bands': 26},
                          kpts={'size': (3, 3, 1), 'gamma': True})
        atoms.get_potential_energy()
        return atoms.calc

    def graphene_pw(self):
        from ase.lattice.hexagonal import Graphene
        atoms = Graphene(symbol='C',
                         latticeconstant={'a': 2.45, 'c': 1.0},
                         size=(1, 1, 1))
        atoms.pbc = (1, 1, 0)
        atoms.center(axis=2, vacuum=4.0)
        ecut = 250
        nkpts = 6
        atoms.calc = GPAW(mode=PW(ecut),
                          kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                          nbands=len(atoms) * 6,
                          txt=self.path / 'graphene_pw.txt')
        atoms.get_potential_energy()
        return atoms.calc

    def mos2_pw(self):
        from ase.build import mx2
        atoms = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                    size=(1, 1, 1), vacuum=5)
        atoms.pbc = (1, 1, 1)
        ecut = 250
        nkpts = 6
        atoms.calc = GPAW(mode=PW(ecut),
                          xc='LDA',
                          kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                          occupations=FermiDirac(0.01),
                          txt=self.path / 'mos2_pw.txt')

        atoms.get_potential_energy()
        return atoms.calc

    def fe_pw(self):
        xc = 'LDA'
        kpts = 4
        nbands = 6
        pw = 300
        occw = 0.01
        conv = {'bands': nbands,
                'density': 1.e-8,
                'forces': 1.e-8}
        a = 2.867
        mm = 2.21
        atoms = bulk('Fe', 'bcc', a=a)
        atoms.set_initial_magnetic_moments([mm])
        atoms.center()

        atoms.calc = GPAW(
            xc=xc,
            mode=PW(pw),
            kpts={'size': (kpts, kpts, kpts)},
            nbands=nbands + 4,
            occupations=FermiDirac(occw),
            convergence=conv,
            txt=self.path / 'fe_pw.txt')

        atoms.get_potential_energy()

        return atoms.calc


class GPAWPlugin:
    def __init__(self):
        if world.rank == -1:
            print()
            info()

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        from gpaw.mpi import size
        terminalreporter.section('GPAW-MPI stuff')
        terminalreporter.write(f'size: {size}\n')


def pytest_configure(config):
    if world.rank != 0:
        try:
            tw = config.get_terminal_writer()
        except AttributeError:
            pass
        else:
            tw._file = devnull
    config.pluginmanager.register(GPAWPlugin(), 'pytest_gpaw')
    for line in ['soc: Spin-orbit coupling',
                 'slow: slow test',
                 'fast: fast test',
                 'ci: test included in CI',
                 'libxc: LibXC requirered',
                 'mgga: MGGA test',
                 'dscf: Delta-SCF',
                 'mom: MOM',
                 'gllb: GLLBSC tests',
                 'elph: Electron-phonon',
                 'intel: fails on INTEL toolchain',
                 'response: tests of the response code',
                 'kspair: tests of kspair in the response code',
                 'serial: run in serial only',
                 'gpu: GPU test',
                 'later: know failure for new refactored GPAW',
                 'legacy: Old stuff that will be removed later',
                 'wannier: Wannier functions']:
        config.addinivalue_line('markers', line)


def pytest_runtest_setup(item):
    """Skip some tests.

    If:

    * they depend on libxc and GPAW is not compiled with libxc
    * they are before $PYTEST_START_AFTER
    """
    from gpaw import libraries

    if world.size > 1:
        for mark in item.iter_markers():
            if mark.name == 'serial':
                pytest.skip('Only run in serial')

    if item.location[0] <= os.environ.get('PYTEST_START_AFTER', ''):
        pytest.skip('Not after $PYTEST_START_AFTER')
        return

    if libraries['libxc']:
        return

    if any(mark.name in {'libxc', 'mgga'}
           for mark in item.iter_markers()):
        pytest.skip('No LibXC.')


@pytest.fixture
def scalapack():
    """Skip if not compiled with sl.

    This fixture otherwise does not return or do anything."""
    from gpaw.utilities import compiled_with_sl
    if not compiled_with_sl():
        pytest.skip(reason='no scalapack')
