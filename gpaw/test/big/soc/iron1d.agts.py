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
    K1 = soc_eigenstates(atoms.calc)
    atoms.calc.write('Fe.gpw')
    K2 = soc_eigenstates('Fe.gpw')
    #occupations={'name': 'fermi-dirac', 'width': 0.05})
    #assert abs(s1['fermi_level'] - s2['fermi_level']) < 1e-10
    for k1, k2 in zip(K1, K2):
        assert abs(k1.eps_n - k2.eps_n).max() < 1e-10
        assert abs(k1.spin_projections_vn -
                   k2.spin_projections_vn).max() < 1e-10
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
    for k1, k2 in zip(A, B):
        assert abs(k1.eps_n - k2.eps_n).max() < 0.003
        assert abs(k1.spin_projections_vn -
                   k2.spin_projections_vn).max() < 0.15


def create_tasks():
    from myqueue.task import task
    return [task('iron1d.agts.py@run', cores=4)]
