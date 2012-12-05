from ase.structure import molecule
from ase.data.vdw import vdw_radii
from gpaw import GPAW
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.contributions import (
    MODE_RADII_CUTOFF, MODE_SURFACE_TENSION, MODE_AREA_VOLUME_VDW
    )
from gpaw.solvation.poisson import SolvationPoissonSolver
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian

def test_save_load(calc):
    solvation = calc.input_parameters['solvation']
    fname = 'solvation_io.gpw'
    calc.write(fname)
    read_calc = SolvationGPAW(fname, txt=None)
    assert isinstance(read_calc.hamiltonian, SolvationRealSpaceHamiltonian)
    assert isinstance(
        read_calc.hamiltonian.poisson, SolvationPoissonSolver
        ), 'Expected %s, got %s' \
        % (SolvationPoissonSolver, read_calc.hamiltonian.poisson)
    assert read_calc.input_parameters['solvation'] == solvation
    for attr in ('Etot', 'Eel', 'Erep', 'Edis', 'Ecav', 'Etm', 'Acav', 'Vcav'):
        assert getattr(calc.hamiltonian, attr) == \
               getattr(read_calc.hamiltonian, attr)
    assert read_calc.get_potential_energy() == calc.get_potential_energy()

h = .3

water = molecule('H2O')
water.center(3.0)
water.calc = GPAW(xc='PBE', h=h)
E = water.get_potential_energy()
F = water.get_forces()

water.calc = SolvationGPAW(
    xc='PBE', h=h,
    solvation={
        'el':{
            'mode':MODE_RADII_CUTOFF,
            'radii':[vdw_radii[a.number] for a in water],
            'exponent':5.2
            },
        'cav':{
            'mode':MODE_SURFACE_TENSION,
            'surface_tension': .0
            },
        'dis':{
            'mode':MODE_AREA_VOLUME_VDW,
            'surface_tension': .0,
            'pressure': .0
            }
        }
    )
Esol = water.get_potential_energy()
Fsol = water.get_forces()
assert Esol == E
assert (F == Fsol).all()
assert Esol == water.calc.get_electrostatic_energy()
assert water.calc.get_repulsion_energy() == .0
assert water.calc.get_dispersion_energy() == .0
assert water.calc.get_cavity_formation_energy() == .0
assert water.calc.get_thermal_motion_energy() == .0

# XXX move to a different test? (then the calculation has to be run twice)
test_save_load(water.calc)
