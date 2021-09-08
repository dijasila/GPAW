class WaveFunctions:
    def __init__(self, wave_functions, spin, setups):
        self.wave_functions = wave_functions
        self.spin = spin
        self.setups = setups

    def orthonormalize(self, projectors, work_matrix, work_array):
        projections = projectors.integrate(self.wave_functions)

        layout = self.wave_functions.layout
        S = work_matrix
        W = self.wave_functions.as_matrix()
        P = projections.as_matrix()

        S[:] = layout.dv * W @ W.C
        P2 = P @ self.setups.dS
        S += P.multiply(P2.C, symmetric=True)
        #cc?

        S.invcholesky(layout.comm)
        # S now contains the inverse of the Cholesky factorization

        W2 = W.new(data=work_array)
        W2[:] = S @ W
        P2[:] = S @ P
        W[:] = W2
        P[:] = P2

        return projections
