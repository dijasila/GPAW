import numpy as np

from .overlaps import WannierOverlaps


def write_win(name: str,
              overlaps: WannierOverlaps,
              **kwargs) -> None:
    kwargs['num_bands'] = overlaps.nbands
    kwargs['fermi_energy'] = overlaps.fermi_level
    kwargs['unit_cell_cart'] = overlaps.atoms.cell
    kwargs['atoms_frac'] = [[symbol] + list(spos_c)
                            for symbol, spos_c
                            in zip(overlaps.atoms.symbols,
                                   overlaps.atoms.get_scaled_positions())]
    kwargs['mp_grid'] = overlaps.monkhorst_pack_size
    with open(f'{name}.win', 'w') as fd:
        for key, val in kwargs.items():
            if isinstance(val, tuple):
                print(f'{key} =', *val, file=fd)
            elif isinstance(val, (list, np.ndarray)):
                print(f'begin {key}', file=fd)
                for line in val:
                    print(*line, file=fd)
                print(f'end {key}', file=fd)
            else:
                print(f'{key} = {val}', file=fd)


"""
def write_mmn(name: str,
              overlaps: WannierOverlaps,
              **kwargs) -> None:
    with open(f'{name}.mmn', 'w') as fd:
        print('Input generated from GPAW', file=fd)
        print('%10d %6d %6d' % (Nn, Nk, Nb), file=f)
        print('%3d %3d %4d %3d %3d' % indices, file=f)
        for m1 in range(len(M_mm)):
            for m2 in range(len(M_mm)):
                M = M_mm[m2, m1]
                print('%20.12f %20.12f' % (M.real, M.imag), file=f)
"""
