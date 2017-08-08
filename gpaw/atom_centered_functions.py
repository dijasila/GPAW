import numpy as np


class AtomCenteredFunctions:
    dtype = float

    def __init__(self, desc, functions_a, kd=None, dtype=float,
                 integral=None, cut=False,
                 comm=None, blocksize=None):
        if desc.__class__.__name__ == 'PWDescriptor':
            self.space = 'reciprocal'
            from gpaw.wavefunctions.pw import PWLFC
            self.lfc = PWLFC(desc, functions_a, blocksize=blocksize, comm=comm)
        else:
            self.space = 'real'
            from gpaw.lfc import LFC
            self.lfc = LFC(desc, functions_a, kd=kd, dtype=dtype,
                           integral=integral, cut=cut)
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
            kpt = array.kpt
            array = array.array
        else:
            kpt = -1

        if not isinstance(coefs, (float, dict)):
            coefs = {a: P_in.T for a, P_in in coefs.items()}

        if self.space == 'reciprocal' and force_real_space:
            pd = self.lfc.pd
            array += pd.ifft(coefs / pd.gd.dv * self.lfc.expand(-1).sum(0))
        else:
            self.lfc.add(array, coefs, kpt)

    def derivative(self, array, out=None):
        if out is None:
            out = {a: np.empty((I2 - I1, 3)) for a, I1, I2 in self.indices}
        self.lfc.derivative(array, out)
        return out

    def integrate(self, array, out=None):
        if out is None:
            out = {a: np.empty(I2 - I1) for a, I1, I2 in self.indices}
        self.lfc.integrate(array, out)
        return out

    def stress_tensor_contribution(self, a_q, b_ax=1.0, q=-1):
        return self.lfc.stress_tensor_contribution(a_q, b_ax, q)

    def eval(self, out):
        out.matrix.array[:] = 0.0
        coef_M = np.zeros(len(self), out.dtype)
        for M, a in enumerate(out.matrix.array):
            coef_M[M] = 1.0
            self.lfc.lfc.lcao_to_grid(coef_M, a, out.kpt)
            coef_M[M] = 0.0

    def matrix_elements(self, other, out, hermetian=False, derivative=False):
        if derivative:
            if out is None:
                N = other.myshape[0]
                out = out = {a: np.empty((N, I2 - I1, 3), other.dtype)
                             for a, I1, I2 in self.indices}
            self.lfc.derivative(other.array, out, other.kpt)
            return out

        self.lfc.integrate(other.array, self.dictview(out.matrix), other.kpt)

    def dictview(self, matrix):
        M_In = matrix.array
        M_ani = {}
        for a, I1, I2 in self.indices:
            M_ani[a] = M_In[I1:I2].T
        return M_ani
