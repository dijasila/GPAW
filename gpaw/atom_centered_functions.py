import numpy as np


class AtomCenteredFunctions:
    dtype = float

    def __init__(self, desc, functions_a, kd=None, integral=None, cut=False,
                 comm=None, blocksize=None):
        if desc.__class__.__name__ == 'PWDescriptor':
            from gpaw.wavefunctions.pw import PWLFC
            self.lfc = PWLFC(desc, functions_a, blocksize=blocksize, comm=comm)
        else:
            from gpaw.lfc import LFC
            self.lfc = LFC(desc, functions_a, integral=integral, cut=cut)
        self.atom_indices = []
        self.slices = []
        I1 = 0
        for a, functions in enumerate(functions_a):
            I2 = I1 + sum(f.l * 2 + 1 for f in functions)
            self.atom_indices.append(a)
            self.slices.append((I1, I2))
            I1 = I2
        self.nfuncs = I1
        self.mynfuncs = I1

    def __len__(self):
        return self.nfuncs

    def set_positions(self, spos_ac, rank_a=None):
        self.lfc.set_positions(spos_ac)

    def add_to(self, array, coefs=1.0):
        if not isinstance(array, np.ndarray):
            array = array.array
        if not isinstance(coefs, (float, dict)):
            coefs = coefs.todict()
        self.lfc.add(array, coefs)

    def integrate(self, array, out=None):
        if out is None:
            out = {a: np.empty(I2 - I1)
                   for a, (I1, I2) in zip(self.atom_indices, self.slices)}
        self.lfc.integrate(array, out)
        return out

    def eval(self, out):
        out.array[:] = 0.0
        coef_M = np.zeros(len(self))
        for M, a in enumerate(out.array):
            coef_M[M] = 1.0
            self.lfc.lfc.lcao_to_grid(coef_M, a, -1)
            coef_M[M] = 0.0

    def matrix_elements(self, other, out, hermetian=False):
        self.lfc.integrate(other.array, self.dictview(out), -1)

    def dictview(self, matrix):
        M_In = matrix.array
        M_ani = {}
        for a, (I1, I2) in zip(self.atom_indices, self.slices):
            M_ani[a] = M_In[I1:I2].T
        return M_ani
