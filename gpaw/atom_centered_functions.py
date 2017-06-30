import numpy as np


class AtomCenteredFunctions:
    dtype = float

    def __init__(self, desc, functions_a, kd=None, integral=None, cut=False,
                 comm=None, blocksize=None):
        if desc.__class__.__name__ == 'PWDescriptor':
            self.space = 'reciprocal'
            from gpaw.wavefunctions.pw import PWLFC
            self.lfc = PWLFC(desc, functions_a, blocksize=blocksize, comm=comm)
        else:
            self.space = 'real'
            from gpaw.lfc import LFC
            self.lfc = LFC(desc, functions_a, integral=integral, cut=cut)
        self.indices = []
        I1 = 0
        for a, functions in enumerate(functions_a):
            I2 = I1 + sum(f.l * 2 + 1 for f in functions)
            self.indices.append((a, I1, I2))
            I1 = I2
        self.nfuncs = I1
        self.mynfuncs = I1

    def __len__(self):
        return self.nfuncs

    def set_positions(self, spos_ac, rank_a=None):
        self.lfc.set_positions(spos_ac)

    def add_to(self, array, coefs=1.0, force_real_space=False):
        if not isinstance(array, np.ndarray):
            array = array.array
        if not isinstance(coefs, (float, dict)):
            coefs = coefs.todict()
        if self.space == 'reciprocal' and force_real_space:
            pd = self.lfc.pd
            array += pd.ifft(coefs / pd.gd.dv * self.lfc.expand(-1).sum(0))
        else:
            self.lfc.add(array, coefs)

    def derivativeeeee(self, a, out=None):
        if out is None:
            out = {a: np.empty(I2 - I1)
                   for a, (I1, I2) in zip(self.atom_indices, self.slices)}
        1 / 0  # PWLFC.derivative(self, self.pd.fft(dedtaut_R), dF_aiv)

    def integrate(self, array, out=None):
        if out is None:
            out = {a: np.empty(I2 - I1) for a, I1, I2 in self.indices}
        self.lfc.integrate(array, out)
        return out

    def eval(self, out):
        out.array[:] = 0.0
        coef_M = np.zeros(len(self))
        for M, a in enumerate(out.array):
            coef_M[M] = 1.0
            self.lfc.lfc.lcao_to_grid(coef_M, a, -1)
            coef_M[M] = 0.0

    def matrix_elements(self, other, out, hermetian=False, derivative=False):
        if derivative:
            if out is None:
                N = other.myshape[0]
                out = out = {a: np.empty((N, I2 - I1, 3), other.dtype)
                             for a, I1, I2 in self.indices}
            self.lfc.derivative(other.array, out)
            return out

        self.lfc.integrate(other.array, self.dictview(out), -1)

    def dictview(self, matrix):
        M_In = matrix.array
        M_ani = {}
        for a, I1, I2 in self.indices:
            M_ani[a] = M_In[I1:I2].T
        return M_ani
