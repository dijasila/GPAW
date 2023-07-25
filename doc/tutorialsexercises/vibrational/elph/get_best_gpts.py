"""
When two separate simulations using different calculators need to use the
same real space grid, like in the electron-phonon coupling case, it might
be necessary to specify gpts=(...) explicitly. This tool helps with finding
the correct values for a combination of PW and LCAO mode calculations.
"""
from typing import List

from gpaw.utilities.gpts import obtain_gpts_suggestion


def main(argv: List[str] = None) -> None:
    import argparse
    from ase.io import read

    parser = argparse.ArgumentParser(
        prog='python3 get_best_gpts.py',
        description='Calculate optimal gpts size between PW and LCAO mode.')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='ASE readable structure file.')
    add('-e', '--ecut', type=float,
        help='Cutoff "ecut" used in PW mode.')
    add('-g', '--grid-spacing', type=float,
        help='Maximal grid spacing "h" used in LCAO mode.')
    add('-s', '--super-cell', default='1,1,1',
        help='Supercell size.  Example: "-s 2,2,2".')

    args = parser.parse_intermixed_args(argv)
    if not args.ecut or not args.grid_spacing:
        parser.print_help()
        raise SystemExit

    sc = []
    for value in args.super_cell.split(','):
        sc.append(int(value))

    atoms = read(args.file)
    atoms_sc = atoms * sc
    cell_cv = atoms_sc.get_cell()
    obtain_gpts_suggestion(cell_cv, args.ecut, args.grid_spacing, True)


if __name__ == "__main__":
    main()
