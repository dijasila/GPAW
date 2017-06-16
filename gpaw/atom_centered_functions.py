import numpy as np

from gpaw.lfc import LFC


class AtomCenteredFunctions:
    dtype = float

    def __init__(self, desc, functions, kd):
        self.lfc = LFC(desc, functions)
        self.atom_indices = []
        self.slices = []
        I1 = 0
        for a, f in enumerate(functions):
            I2 = I1 + len(f)
            self.atom_indices.append(a)
            self.slices.append((I1, I2))
            I1 = I2
        self.nfuncs = I2

    def __len__(self):
        return self.nfuncs

    def set_positions(self, spos_ac, rank_a=None):
        self.lfc.set_positions(spos_ac)

    def eval(self, out):
        out.array[:] = 0.0
        coef_M = np.zeros(len(self))
        for M, a in enumerate(out.array):
            coef_M[M] = 1.0
            self.lfc.lfc.lcao_to_grid(coef_M, a, -1)
            print(M, a.ptp())
            coef_M[M] = 0.0

    def matrix_elements(self, other, out, hermetian=False):
        self.lfc.integrate(other.array, self.dictview(out), -1)

    def dictview(self, matrix):
        M_In = matrix.array
        M_ani = {}
        for a, (I1, I2) in zip(self.atom_indices, self.slices):
            M_ani[a] = M_In[I1:I2].T
        return M_ani
