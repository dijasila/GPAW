from typing import Any

from ase import Atoms

from gpaw import GPAW

Array2D = Any
Array3D = Any


class WannierFunctions:
    def __init__(self,
                 atoms: Atoms,
                 U_nn,
                 centers,
                 value,
                 n1=0,
                 spin=0):
        self.atoms = atoms
        self.U_nn = U_nn
        self.centers = centers
        self.value = value
        self.n1 = 0
        self.spin = spin

    def get_function(self, calc: GPAW, n: int) -> Array3D:
        wf = 0.0
        for m, u in enumerate(self.U_nn[:, n]):
            wf += u * calc.wfs.get_wave_function_array(n=self.n1 + m,
                                                       s=self.spin,
                                                       k=0)
        return wf

    def centers_as_atoms(self):
        return self.atoms + Atoms(f'X{len(self.centers)}', self.centers)
