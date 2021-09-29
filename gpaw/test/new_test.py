from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import GPAW as NewGPAW
from ase import Atoms


def new(q):
    atoms = Atoms('H2')
    atoms.positions[1, 2] = 0.8
    atoms.center(vacuum=1)
    params = {'txt': '-',
              'gpts': (8, 8, 12),
              'random': 1}
    import matplotlib.pyplot as plt
    if 'n' in q:
        atoms.calc = NewGPAW(**params)
        atoms.get_potential_energy()
        # w = atoms.calc.get_pseudo_wave_function(0)
        # print(w[4, 4])
        # x, y = atoms.calc.calculation.density.density.xy(0, 4, 4, ...)
        x, y = atoms.calc.calculation.potential.vt.xy(0, 4, 4, ...)
        plt.plot(x, y)
        x, y = atoms.calc.calculation.scf.pot_calc.v0.xy(8, 8, ...)
        plt.plot(x, y)
    if 'o' in q:
        atoms.calc = OldGPAW(**params)
        atoms.get_potential_energy()
        x = atoms.calc.density.gd.coords(2, pad=False)
        # plt.plot(x, atoms.calc.density.nt_sG[0, 4, 4])
        plt.plot(x, atoms.calc.hamiltonian.vt_sG[0, 4, 4])
    plt.show()


if __name__ == '__main__':
    import sys
    new(sys.argv[1])
