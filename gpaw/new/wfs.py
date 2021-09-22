class IBZWaveFunctions:
    def __init__(self, ibz, ranks, kpt_comm, mykpts):
        self.ibz = ibz
        self.ranks = ranks
        self.kpt_comm = kpt_comm
        self.mykpts = mykpts

    @classmethod
    def from_random_numbers(self, base, nbands):
        ibz = base.ibz
        assert len(ibz) == 1
        ranks = [0]
        band_comm = base.communicators['b']
        kpt_comm = base.communicators['k']

        mykpts = []
        for kpt, weight, rank in zip(ibz.points, ibz.weights, ranks):
            if rank != kpt_comm.rank:
                continue
            basis = base.grid.new(kpt=kpt)
            wfs = WaveFunctions.from_random_numbers(basis, weight,
                                                    nbands, band_comm,
                                                    base.setups, base.positions)
            mykpts.append(wfs)

        return IBZWaveFunctions(ibz, ranks, kpt_comm, mykpts)


class WaveFunctions:
    def __init__(self, wave_functions, spin, setups, positions):
        self.wave_functions = wave_functions
        self.spin = spin
        self.setups = setups
        self._projections = None
        self.projectors = setups.create_projectors(wave_functions.layout,
                                                   positions)

    @property
    def projections(self):
        if self._projections is None:
            self._projections = self.projectors.matrix_element(
                self.wave_functions)
        return self._projections

    @classmethod
    def from_random_numbers(self, basis, weight, nbands, band_comm, setups,
                            positions):
        wfs = basis.random(nbands, band_comm)
        return WaveFunctions(wfs, 0, setups, positions)

    def orthonormalize(self, work_array=None):
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
