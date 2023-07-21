import numpy as np

from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.wavefunctions.pw import PW
from gpaw.wavefunctions.lcao import LCAO


def obtain_gpts_suggestion(atoms, ecut, h, lprint=False):
    """Compare PW and LCAO gpts and returns tighter one.

    Parameters
    ----------
    atoms: Atoms
        Atoms object to be used
    ecut: int, float
        planewave cutoff to be used
    h: float
        intended maximal grid spacing
    lprint: bool
       if True, prints human readable information
    """

    cell_cv = atoms.get_cell()

    Npw_c = get_number_of_grid_points(cell_cv, mode=PW(ecut))
    Nlcao_c = get_number_of_grid_points(cell_cv, h=h, mode=LCAO())

    # Nopt_c = np.maximum(Npw_c, Nlcao_c)
    Nopt_c = np.maximum(Nlcao_c, (Npw_c / 4 + 0.5).astype(int) * 4)

    if lprint:
        print(f"PW({ecut:3.0f}) -> gpts={list(Npw_c)}")
        print(f"LCAO, h={h:1.3f} -> gpts={list(Nlcao_c)}")
        print(f"Recommended for elph: gpts={list(Nopt_c)}")

        if np.all(Npw_c == Nlcao_c):
            print("  Both sets of gpts the same. No action needed.")
        if np.any(Npw_c < Nopt_c):
            print(f"  Add 'gpts={list(Nopt_c)}' to PW mode calculator.")
        if np.any(Nlcao_c < Nopt_c):
            print(f"  Use 'gpts={list(Nopt_c)}' instead of 'h={h:1.3f}' \
                  in LCAO calculator.")

    return Nopt_c


def main():
    from sys import argv
    from ase.io import read

    # get stuff from shell
    if not len(argv) in [4, 7]:
        print("Usage: python get_best_gpts.py atomsfile ecut h \
            supercellfactors")
        print("where supercellfactors are optional. \
            If used need to be 3 space separated integers")
        exit()

    atoms_file = argv[1]  # crystal structure
    ecut = float(argv[2])  # planewave cutoff
    h = float(argv[3])  # desired LCAO grid spacing
    if len(argv) == 7:
        sc = []
        for i in range(4, 7):
            sc.append(int(argv[i]))
    else:
        sc = [1, 1, 1]

    atoms = read(atoms_file)
    atoms_sc = atoms * sc

    obtain_gpts_suggestion(atoms_sc, ecut, h, True)


if __name__ == "__main__":
    main()
