class WaveFunctions:
    def __init__(self, wave_functions, spin, setups):
        self.wave_functions = wave_functions
        self.spin = spin
        self.setups = setups

    def orthonormalize(self, projectors, work_matrix, work_array):
        wfs = self.wave_functions
        domain_comm = wfs.layout.comm

        projections = projectors.integrate(wfs)

        S = work_matrix
        projections2 = projections.new()
        wfs2 = wfs.new(data=work_array)

        def dS(proj):
            for a, I1, I2 in proj.layout.myindices:
                ds = self.setups[a].dO_ii
                # use mmm ?????
                projections2.data[I1:I2] = ds @ proj.data[I1:I2]
            return projections2

        wfs.matrix_elements(wfs, domain_sum=False, out=S)
        projections.matrix_elements(projections, function=dS,
                                    domain_sum=False, out=S, add_to_out=True)

        domain_comm.sum(S.data, 0)
        if domain_comm.rank == 0:
            S.invcholesky()
        # S now contains the inverse of the Cholesky factorization
        domain_comm.broadcast(S.data, 0)
        # cc ??????

        S.multiply(wfs, out=wfs2)
        projections.matrix.multiply(S, opb='T', out=projections2)
        wfs.data[:] = wfs2.data
        projections.data[:] = projections2.data

        return projections
