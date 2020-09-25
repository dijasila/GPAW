import subprocess
from pathlib import Path
from typing import Union

from ase import Atoms
import numpy as np

from .overlaps import WannierOverlaps
from .functions import WannierFunctions


class Wannier90:
    def __init__(self,
                 prefix: str = 'wannier',
                 folder: Union[str, Path] = 'W90',
                 executable='wannier90.x'):
        self.prefix = prefix
        self.folder = Path(folder)
        self.executable = executable
        self.folder.mkdir(exist_ok=True)

    def run_wannier90(self, postprocess=False, world=None):
        args = [self.executable, self.prefix]
        if postprocess:
            args[1:1] = ['-pp']
        result = subprocess.run(args, cwd=self.folder)
        print(result)

    def write_input_files(self,
                          overlaps: WannierOverlaps,
                          **kwargs) -> None:
        self.write_win(overlaps, **kwargs)
        self.write_mmn(overlaps)
        self.write_amn(overlaps)

    def write_win(self,
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

        with (self.folder / f'{self.prefix}.win').open('w') as fd:
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

    def write_mmn(self,
                  overlaps: WannierOverlaps) -> None:
        nbzkpts, nwan, nbands = overlaps.projections.shape
        size = overlaps.monkhorst_pack_size
        assert np.prod(size) == nbzkpts

        with (self.folder / f'{self.prefix}.mmn').open('w') as fd:
            print('Input generated from GPAW', file=fd)
            print(f'{nwan} {nbzkpts} {nbands}', file=fd)

            for bz_index1 in range(nbzkpts):
                i1_c = np.unravel_index(bz_index1, size)
                for direction, d in overlaps.directions.items():
                    i2_c = np.array(i1_c) + direction
                    bz_index2 = np.ravel_multi_index(i2_c, size, 'wrap')
                    a, b, c = (i2_c % size - i2_c) // size
                    print(f'{bz_index1} {bz_index2} {a} {b} {c}', file=fd)
                    M_nn = overlaps.overlap(bz_index1, direction)
                    for M_n in M_nn:
                        for M in M_n:
                            print(f'{M.real} {M.imag}', file=fd)

    def write_amn(self,
                  overlaps: WannierOverlaps) -> None:
        proj_kmn = overlaps.projections
        nbzkpts, nwan, nbands = proj_kmn.shape

        with (self.folder / f'{self.prefix}.amn').open('w') as fd:
            print('Input generated from GPAW', file=fd)
            print(f'{nbands} {nbzkpts} {nwan}', file=fd)

            for bz_index, proj_mn in enumerate(proj_kmn):
                for m, proj_n in enumerate(proj_mn):
                    for n, P in enumerate(proj_n):
                        print(n + 1, m + 1, bz_index + 1, P.real, P.imag,
                              file=fd)

    def read_result(self):
        return Wannier90Functions()


class Wannier90Functions(WannierFunctions):
    def __init__(self,
                 atoms: Atoms,
                 centers):
        WannierFunctions.__init__(self, atoms, centers)
