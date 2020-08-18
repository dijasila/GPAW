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
    name = 'Fe-sym-' + params.get('symmetry', 'on')
    atoms.calc = GPAW(txt=name + '.txt', **params)
    atoms.get_potential_energy()
    s1 = soc_eigenstates(atoms.calc)
    atoms.calc.write(name + '.gpw')
    s2 = soc_eigenstates(name + '.gpw')
    assert abs(s1.fermi_level - s2.fermi_level) < 1e-10
    assert abs(s1.eigenvalues() - s2.eigenvalues()).max() < 1e-10
    assert abs(s1.spin_projections() - s2.spin_projections()).max() < 1e-7
    for wf1, wf2 in zip(s1, s2):
        P1_msI = wf1.projections.array
        P2_msI = wf2.projections.array
        assert abs(P1_msI - P2_msI).max() < 1e-7
    return s1, atoms.calc


def run() -> None:
    """Compare with and without symmetry."""
    params = dict(mode=PW(400),
                  xc='PBE',
                  kpts=[7, 1, 1])
    A, _ = soc(params)
    params['symmetry'] = 'off'
    B, calc = soc(params)
    assert abs(A.fermi_level - B.fermi_level) < 0.002
    assert abs(A.eigenvalues() - B.eigenvalues()).max() < 0.003
    assert abs(A.spin_projections() - B.spin_projections()).max() < 0.15

    for wf in B:
        assert wf.wavefunctions(calc).shape == (18, 2, 12, 24, 24)


def create_tasks():
    from myqueue.task import task
    return [task('iron1d_agts.py', cores=4)]


if __name__ == '__main__':
    run()
