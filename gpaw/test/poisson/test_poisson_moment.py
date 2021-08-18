import numpy as np
import pytest

from ase.units import Bohr
from gpaw.poisson import FDPoissonSolver, NoInteractionPoissonSolver
from gpaw.poisson_moment import MomentCorrectionPoissonSolver
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver
from gpaw.grid_descriptor import GridDescriptor

from gpaw.test import equal


@pytest.mark.parametrize('moment_corrections, expected_len', [
    (None, 0),
    ([], 0),
    (4, 1),
    (9, 1),
    ([dict(moms=range(4), center=np.array([1, 3, 5]))], 1),
    ([dict(moms=range(4), center=np.array([5, 3, 5])),
     dict(moms=range(4), center=np.array([7, 5, 3]))], 2)
])
def test_defaults(moment_corrections, expected_len):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    assert isinstance(poisson.moment_corrections, list), \
        poisson.moment_corrections
    assert len(poisson.moment_corrections) == expected_len
    assert all([isinstance(mom, dict) for mom in poisson.moment_corrections])


@pytest.mark.parametrize('moment_corrections', [
    (None),
    ([]),
])
def test_description_empty(moment_corrections):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    desc = poisson.get_description()
    desc_ref = poisson_ref.get_description()

    assert isinstance(desc, str)
    assert isinstance(desc_ref, str)
    assert desc_ref in desc
    assert '0 moment corrections' in desc


@pytest.mark.parametrize('moment_corrections, expected_strings', [
    (4, ['1 moment corrections', 'center', 'range(0, 4)']),
    (9, ['1 moment corrections', 'center', 'range(0, 9)']),
    ([dict(moms=range(4), center=np.array([1, 1, 1]))],
        ['1 moment corrections', '[1.00, 1.00, 1.00]', 'range(0, 4)']),
    ([dict(moms=[1, 2, 3], center=np.array([1, 1, 1]))],
        ['1 moment corrections', '[1.00, 1.00, 1.00]', 'range(1, 4)']),
    ([dict(moms=[0, 2, 3], center=np.array([1, 1, 1]))],
        ['1 moment corrections', '[1.00, 1.00, 1.00]', '(0, 2, 3)']),
    ([dict(moms=range(4), center=np.array([2, 3, 4])),
      dict(moms=range(4), center=np.array([7.4, 3.1, 0.1]))],
        ['2 moment corrections', '[2.00, 3.00, 4.00]',
         '[7.40, 3.10, 0.10]', 'range(0, 4)']),
])
def test_description(moment_corrections, expected_strings):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    desc = poisson.get_description()

    desc = poisson.get_description()
    desc_ref = poisson_ref.get_description()

    assert isinstance(desc, str)
    assert isinstance(desc_ref, str)

    # Make sure that the description starts with the description of the wrapped
    # solver
    assert desc.startswith(desc_ref)

    # and follows with the moments
    desc_rem = desc[len(desc_ref):]
    for expected_str in expected_strings:
        assert expected_str in desc_rem, \
            f'"{expected_str}" not in "{desc_rem}"'


@pytest.fixture
def gd():
    N_c = (16, 16, 3 * 16)
    cell_cv = (1, 1, 3)
    gd = GridDescriptor(N_c, cell_cv, False)

    return gd


@pytest.mark.parametrize('moment_corrections', [
    4,
    9,
    [dict(moms=range(4), center=np.array([1, 1, 1]))],
    [dict(moms=range(4), center=np.array([2, 3, 4])),
     dict(moms=range(4), center=np.array([7.4, 3.1, 0.1]))],
])
def test_write(gd, moment_corrections):
    poisson_ref = FDPoissonSolver()
    poisson_ref.set_grid_descriptor(gd)

    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)
    poisson.set_grid_descriptor(gd)

    from gpaw.io import Writer
    from gpaw.mpi import world
    filename = '/dev/null'
    writer = Writer(filename, world)
    writer.child('poisson').write(**poisson.todict())
    writer.close()


def test_poisson_moment_correction(gd):

    # Construct model density
    coord_vg = gd.get_grid_point_coordinates()
    z_g = coord_vg[2, :]
    rho_g = gd.zeros()
    for z0 in [1, 2]:
        rho_g += 10 * (z_g - z0) * \
            np.exp(-20 * np.sum((coord_vg.T - np.array([.5, .5, z0])).T**2,
                                axis=0))

    poissoneps = 1e-20

    do_plot = False

    if do_plot:
        big_rho_g = gd.collect(rho_g)
        if gd.comm.rank == 0:
            import matplotlib.pyplot as plt
            fig, ax_ij = plt.subplots(3, 4, figsize=(20, 10))
            ax_i = ax_ij.ravel()
            ploti = 0
            Ng_c = gd.get_size_of_global_array()
            plt.sca(ax_i[ploti])
            ploti += 1
            plt.pcolormesh(big_rho_g[Ng_c[0] // 2])
            plt.text(0.05, 0.96, 'density', transform=plt.gca().transAxes,
                     ha='left', va='top', color='white')
            plt.sca(ax_i[ploti])
            ploti += 1
            plt.plot(big_rho_g[Ng_c[0] // 2, Ng_c[1] // 2])

    def plot_phi(phi_g, label=None):
        if do_plot:
            big_phi_g = gd.collect(phi_g)
            if gd.comm.rank == 0:
                nonlocal ploti
                plt.sca(ax_i[ploti])
                ploti += 1
                plt.pcolormesh(big_phi_g[Ng_c[0] // 2])
                if label is not None:
                    plt.text(0.05, 0.96, label, transform=plt.gca().transAxes,
                             ha='left', va='top', color='white')
                plt.sca(ax_i[ploti])
                ploti += 1
                plt.plot(big_phi_g[Ng_c[0] // 2, Ng_c[1] // 2])
                plt.ylim(np.array([-1, 1]) * 0.15)

    def poisson_solve(gd, rho_g, poisson):
        phi_g = gd.zeros()
        npoisson = poisson.solve(phi_g, rho_g)
        return phi_g, npoisson

    def compare(phi1_g, phi2_g, val):
        big_phi1_g = gd.collect(phi1_g)
        big_phi2_g = gd.collect(phi2_g)
        if gd.comm.rank == 0:
            equal(np.max(np.absolute(big_phi1_g - big_phi2_g)),
                  val, np.sqrt(poissoneps))

    # Get reference from default poissonsolver
    poisson_ref = FDPoissonSolver(eps=poissoneps)
    poisson_ref.set_grid_descriptor(gd)
    phiref_g, npoisson = poisson_solve(gd, rho_g, poisson_ref)

    # Test agreement with default
    poisson = MomentCorrectionPoissonSolver(poissonsolver=poisson_ref,
                                            moment_corrections=None)
    poisson.set_grid_descriptor(gd)
    phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
    plot_phi(phi_g, 'phi no correction')
    compare(phi_g, phiref_g, 0.0)

    # Test moment_corrections=int
    poisson = MomentCorrectionPoissonSolver(poissonsolver=poisson_ref,
                                            moment_corrections=4)
    poisson.set_grid_descriptor(gd)
    phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
    plot_phi(phi_g, 'phi dipole correction')
    compare(phi_g, phiref_g, 4.1182101206e-02)

    # Test moment_corrections=list
    # Remember that the solver expects Ångström units and we have specified
    # the grid in Bohr
    moment_corrections = [{'moms': range(4),
                           'center': np.array([.5, .5, 1]) * Bohr},
                          {'moms': range(4),
                           'center': np.array([.5, .5, 2]) * Bohr}]
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)
    poisson.set_grid_descriptor(gd)
    phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
    plot_phi(phi_g, 'phi multi dipole correction')
    compare(phi_g, phiref_g, 2.7569628594e-02)

    # Get reference from extravacuum solver
    poisson_ref2 = ExtraVacuumPoissonSolver(gpts=(24, 24, 3 * 24),
                                            poissonsolver_large=poisson_ref)
    poisson_ref2.set_grid_descriptor(gd)
    phiref2_g, npoisson = poisson_solve(gd, rho_g, poisson_ref2)

    # extravacuum + no correction
    poisson = MomentCorrectionPoissonSolver(poissonsolver=poisson_ref2,
                                            moment_corrections=None)
    poisson.set_grid_descriptor(gd)
    phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
    compare(phi_g, phiref2_g, 0.0)
    plot_phi(phi_g, 'phi extra vacuum no correction')

    # extravacuum + multi dipole correction
    moment_corrections = [{'moms': range(4),
                           'center': np.array([.5, .5, 1]) * Bohr},
                          {'moms': range(4),
                           'center': np.array([.5, .5, 2]) * Bohr}]
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref2,
        moment_corrections=moment_corrections)
    poisson.set_grid_descriptor(gd)
    phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
    plot_phi(phi_g, 'phi extra vacuum + multi dipole correction')
    compare(phi_g, phiref_g, 2.7523450309062e-02)

    if do_plot:
        if gd.comm.rank == 0:
            plt.show()
