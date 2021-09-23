import numpy as np
from functools import partial


class IBZWaveFunctions:
    def __init__(self, ibz, ranks, kpt_comm, mykpts):
        self.ibz = ibz
        self.ranks = ranks
        self.kpt_comm = kpt_comm
        self.mykpts = mykpts

    @classmethod
    def from_random_numbers(self, cfg, nbands):
        ibz = cfg.ibz
        assert len(ibz) == 1
        ranks = [0]
        band_comm = cfg.communicators['b']
        kpt_comm = cfg.communicators['k']

        mykpts = []
        for kpt, weight, rank in zip(ibz.points, ibz.weights, ranks):
            if rank != kpt_comm.rank:
                continue
            basis = cfg.grid.new(kpt=kpt)
            wfs = WaveFunctions.from_random_numbers(basis, weight,
                                                    nbands, band_comm,
                                                    cfg.setups,
                                                    cfg.positions)
            mykpts.append(wfs)

        return IBZWaveFunctions(ibz, ranks, kpt_comm, mykpts)

    def orthonormalize(self, work_array=None):
        for wfs in self.mykpts:
            wfs.orthonormalize(work_array)


class WaveFunctions:
    def __init__(self, wave_functions, spin, setups, positions):
        self.wave_functions = wave_functions
        self.spin = spin
        self.setups = setups
        self._projections = None
        self.projectors = setups.create_projectors(wave_functions.layout,
                                                   positions)
        self.orthonormalized = False
        self.eps_n = None

    @property
    def projections(self):
        if self._projections is None:
            self._projections = self.projectors.integrate(self.wave_functions)
        return self._projections

    @classmethod
    def from_random_numbers(self, basis, weight, nbands, band_comm, setups,
                            positions):
        wfs = basis.random(nbands, band_comm)
        return WaveFunctions(wfs, 0, setups, positions)

    def orthonormalize(self, work_array=None):
        if self.orthonormalized:
            return
        wfs = self.wave_functions
        domain_comm = wfs.layout.comm

        projections = self.projections

        projections2 = projections.new()
        wfs2 = wfs.new(data=work_array)

        def dS(proj):
            for a, I1, I2 in proj.layout.myindices:
                ds = self.setups[a].dO_ii
                # use mmm ?????
                projections2.data[I1:I2] = ds @ proj.data[I1:I2]
            return projections2

        S = wfs.matrix_elements(wfs, domain_sum=False)
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

        self.orthonormalized = True

    def subspace_diagonalize(self,
                             Ht,
                             dH,
                             work_array=None,
                             Htpsit=None,
                             scalapack_parameters=(None, 1, 1, -1)):
        """

        Ht(in, out)::

           ~   ^   ~
           H = T + v

        dH::

            ~  ~    a    ~  ~
          <psi|p> dH    <p|psi>
              m i   ij    j   n
        """
        self.orthonormalize(work_array)
        psit = self.wave_functions
        projections = self.projections
        psit2 = psit.new(data=work_array)
        projections2 = projections.new()
        domain_comm = psit.layout.comm

        Ht = partial(Ht, out=psit2)
        H = psit.matrix_elements(psit, function=Ht, domain_sum=False)
        projections.matrix_elements(projections, function=dH,
                                    domain_sum=False, out=H, add_to_out=True)
        domain_comm.sum(H.data, 0)

        if domain_comm.rank == 0:
            slcomm, r, c, b = scalapack_parameters
            if r == c == 1:
                slcomm = None
            self.eps_n = H.eigh(scalapack=(slcomm, r, c, b))
            # H.data[n, :] now contains the n'th eigenvector and eps_n[n]
            # the n'th eigenvalue

        domain_comm.broadcast(H.data, 0)
        domain_comm.broadcast(self.eps_n, 0)

        if Htpsit is not None:
            H.multiply(psit2, out=Htpsit)

        H.multiply(psit, out=psit2)
        psit.data[:] = psit2.data
        projections.matrix.multiply(H, opb='T', out=projections2)
        projections.data[:] = projections2.data
