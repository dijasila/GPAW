from ase.build import molecule

from gpaw import GPAW, FermiDirac, PoissonSolver
from gpaw.test import equal, gen

# Generate setup for oxygen with a core-hole:


def test_corehole_h2o_dks():
    def xc(name):
        return {'name': name, 'stencil': 1}

    s = gen('O', name='fch1s', xcname='PBE', corehole=(1, 0, 1.0))

    atoms = molecule('H2O')
    atoms.center(vacuum=2.5)

    calc = GPAW(xc=xc('PBE'),
                poissonsolver=PoissonSolver('fd',
                                            use_charge_center=True))
    atoms.calc = calc
    e1 = atoms.get_potential_energy() + calc.get_reference_energy()
    niter1 = calc.get_number_of_iterations()

    atoms[0].magmom = 1
    calc.set(charge=-1,
             setups={'O': s},
             occupations=FermiDirac(0.0, fixmagmom=True))
    e2 = atoms.get_potential_energy() + calc.get_reference_energy()
    niter2 = calc.get_number_of_iterations()

    atoms[0].magmom = 0
    calc.set(charge=0,
             setups={'O': s},
             occupations=FermiDirac(0.0, fixmagmom=True),
             spinpol=True)
    e3 = atoms.get_potential_energy() + calc.get_reference_energy()
    niter3 = calc.get_number_of_iterations()

    print('Energy difference %.3f eV' % (e2 - e1))
    print('XPS %.3f eV' % (e3 - e1))

    print(e2 - e1)
    print(e3 - e1)
    assert abs(e2 - e1 - 533.070) < 0.001
    assert abs(e3 - e1 - 538.549) < 0.001

    energy_tolerance = 0.02
    print(e1, niter1)
    print(e2, niter2)
    print(e3, niter3)
    equal(e1, -2080.3651, energy_tolerance)
    equal(e2, -1547.2944, energy_tolerance)
    equal(e3, -1541.8152, energy_tolerance)


if __name__ == '__main__':
    test_corehole_h2o_dks()
