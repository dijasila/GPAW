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


def check(array1_mx, array2_mx, k):
    m = 0
    for a1_x, a2_x in zip(array1_mx, array2_mx):
        b1_y = a1_x.ravel()
        b2_y = a2_x.ravel()
        y = abs(b1_y).argmax()
        # assert abs(b1_y - b2_y * b1_y[y] / b2_y[y]).max() < 1e-6
        if abs(b1_y - b2_y * b1_y[y] / b2_y[y]).max() > 1e-6:
            print('J', k, m, abs(b1_y - b2_y * b1_y[y] / b2_y[y]).max())
        m += 1


def soc(params: Dict) -> list:
    """Do DFT + SOC calculations in memory and from gpw-file."""
    name = 'Fe-sym-' + params.get('symmetry', 'on')
    atoms.calc = GPAW(txt=name + '.txt', **params)
    atoms.get_potential_energy()
    s1 = soc_eigenstates(atoms.calc)
    atoms.calc.write(name + '.gpw')
    s2 = soc_eigenstates(name + '.gpw')
    print('F', abs(s1.fermi_level - s2.fermi_level))
    # assert abs(s1.fermi_level - s2.fermi_level) < 1e-10
    print('E', abs(s1.eigenvalues() - s2.eigenvalues()).max())
    assert abs(s1.eigenvalues() - s2.eigenvalues()).max() < 1e-8
    p1 = s1.spin_projections()
    p2 = s2.spin_projections()
    print('SP', abs(p1 - p2).max())
    if atoms.calc.world.rank == 0:
        print('SP', abs(p2 - p1).max(2).max(1))
    # assert abs(p1 - p2).max() < 0.01
    for wf1, wf2 in zip(s1, s2):
        P1_msI = wf1.projections.array
        P2_msI = wf2.projections.array
        check(P1_msI, P2_msI, wf1.bz_index)
    return s1, atoms.calc


def run() -> None:
    """Compare with and without symmetry."""
    params = dict(mode=PW(400),
                  xc='PBE',
                  kpts=[7, 1, 1],
                  convergence={'eigenstates': 1e-10})
    A, _ = soc(params)
    params['symmetry'] = 'off'
    B, calc = soc(params)
    assert abs(A.fermi_level - B.fermi_level) < 0.002
    assert abs(A.eigenvalues() - B.eigenvalues()).max() < 0.003
    p1 = A.spin_projections()
    p2 = B.spin_projections()
    print(p1[1, 1])
    print(p1[5, 1])
    print(p2[1, 1])
    print(p2[5, 1])
    assert abs(A.spin_projections() - B.spin_projections()).max() < 0.15
    for wf1, wf2 in zip(A, B):
        P1_msI = wf1.projections.array
        P2_msI = wf2.projections.array
        check(P1_msI, P2_msI, wf1.bz_index)


def create_tasks():
    from myqueue.task import task
    return [task('iron1d_agts.py', cores=4)]


if __name__ == '__main__':
    run()
