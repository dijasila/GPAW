# MQ: cores=4
"""Check non self-consistent SOC calculation with and without symmetry."""
from typing import Dict
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.spinorbit import soc_eigenstates

atoms = Atoms('Fe',
              magmoms=[2.5],
              cell=[2.5, 5, 5],
              pbc=True)


def soc(params: Dict) -> list:
    """Do DFT + SOC calculations in memory and from gpw-file."""
    atoms.calc = GPAW(**params)
    atoms.get_potential_energy()
    S1 = soc_eigenstates(atoms.calc)
    atoms.calc.write('Fe.gpw')
    S2 = soc_eigenstates('Fe.gpw')
    #occupations={'name': 'fermi-dirac', 'width': 0.05})
    #assert abs(s1['fermi_level'] - s2['fermi_level']) < 1e-10
    assert abs(S1.eigenvalues() - S2.eigenvalues()).max() < 1e-10
    assert abs(S1.spin_projections() - S2.spin_projections()).max() < 0.01
    return K1


def run() -> None:
    """Compare with and without symmetry."""
    params = dict(mode=PW(400),
                  xc='PBE',
                  kpts=[7, 1, 1])
    A = soc(params)
    params['symmetry'] = 'off'
    B = soc(params)
    #assert abs(A['fermi_level'] - B['fermi_level']) < 0.002
    assert abs(A.eigenvalues() - B.eigenvalues()).max() < 0.003
    assert abs(A.spin_projections() - B.spin_projections()).max() < 0.15


def create_tasks():
    from myqueue.task import task
    return [task('iron1d_agts.py@run', cores=4)]
