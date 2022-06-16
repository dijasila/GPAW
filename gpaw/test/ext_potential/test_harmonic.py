import numpy as np
from ase import Atoms
from ase.units import Ha

from gpaw import GPAW
from gpaw.external import ExternalPotential, known_potentials
from gpaw.poisson import NoInteractionPoissonSolver
from gpaw.test import equal


class HarmonicPotential(ExternalPotential):
    def calculate_potential(self, gd):
        a = gd.cell_cv[0, 0]
        r_vg = gd.get_grid_point_coordinates()
        self.vext_g = 0.5 * ((r_vg - a / 2)**2).sum(0)

    def todict(self):
        return {'name': 'HarmonicPotential'}


def test_ext_potential_harmonic(in_tmp_dir):
    """Test againts analytic result (no xc, no Coulomb)."""
    a = 4.0
    x = Atoms(cell=(a, a, a))  # no atoms

    calc = GPAW(charge=-8,
                nbands=4,
                h=0.2,
                xc={'name': 'null'},
                external=HarmonicPotential(),
                poissonsolver=NoInteractionPoissonSolver(),
                eigensolver='cg')

    x.calc = calc
    x.get_potential_energy()

    eigs = calc.get_eigenvalues()
    equal(eigs[0], 1.5 * Ha, 0.002)
    equal(abs(eigs[1:] - 2.5 * Ha).max(), 0, 0.003)

    # Check write + read:
    calc.write('harmonic.gpw')
    known_potentials['HarmonicPotential'] = HarmonicPotential
    GPAW('harmonic.gpw')


class PöchelTellerPotential(ExternalPotential):
    def calculate_potential(self, gd):
        a = gd.cell_cv[2, 2]
        r_vg = gd.get_grid_point_coordinates()
        lam = 1
        self.vext_g = -lam * (lam + 1) / 2 * np.cosh(r_vg[2] - a / 2)**-2

    def todict(self):
        return {'name': 'HarmonicPotential'}


def test_pt_potential(in_tmp_dir):
    """Test againts analytic result (no xc, no Coulomb)."""
    d = 6.0
    a = 2
    x = Atoms(cell=(a, a, d), pbc=[1, 1, 0])  # no atoms

    calc = GPAW(charge=-2,
                nbands=1,
                mode=dict(name='pw', ecut=500),
                xc={'name': 'null'},
                external=PöchelTellerPotential(),
                poissonsolver=NoInteractionPoissonSolver(),
                #eigensolver='cg'
                )

    x.calc = calc
    x.get_potential_energy()

    (eig,) = calc.get_eigenvalues()
    print(eig)
    assert eig == -0.5 * Ha
