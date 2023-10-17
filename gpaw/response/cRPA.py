class cRPA_weight:
    def __init__(self, extra_weights_n):
        self.extra_weights_n = extra_weights_n
        self.extra_pair_weights_nm = self.set_extra_pair_weights()

    @classmethod
    def from_wannier_matrix(cls, Uwan_wnk, bandrange):
        from gpaw.wannier90 import read_uwan

        # if Uwan is string try to read wannier90 matrix
        if isinstance(Uwan_knw, str):
            seed = Uwan_knw
            assert kd is not None
            if "_u.mat" not in seed:
                seed += "_u.mat"
            uwan, nk, nw1, nw2 = read_uwan(seed, self.gs.kd)
        R_asii = calc.setups.atomrotations.get_R_asii()
        return cls(calc.wfs.kd, calc.spos_ac, R_asii, calc.wfs.gd.N_c)


    def set_extra_pair_weights(self):
        nb = len(self.extra_weights_n)
        weights_nm = np.zeros(nb, nb)
        for i in range(nb):
            for j in range(nb):
                self.extra_pair_weights_nm[i, j] = (1.0 - self.extra_weights_n[i]) * # ... Use cRPA formula... XXX write using array ops not loops if possible

        
class cRPA(Chi0Calculator):
    """Class for calculating constrained non-interacting response functions
    """

    def __init__(self,
                 calc,
                 extra_weights,
                 *,
                 frequencies: Union[dict, Array1D] = None,
                 ecut=50,
                 threshold=1,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1,
                 nbands=None,
                 **kwargs):
        """Construct Chi0 object.

        Parameters
        ----------
        extra_weights: a list of weights for the model subspace bands. 1=included
                in model. Can also be noninteger and computed from e.g Wannier
                transformation matrices. Sould have the same length as the number
                bands in the calculation.
        Remaining parameters: See Chi0

        """
        from gpaw.response.pair import get_gs_and_context
        gs, context = get_gs_and_context(calc, txt, world, timer)
        nbands = nbands or gs.bd.nbands

        wd = new_frequency_descriptor(
            gs, context, nbands, frequencies,
            domega0=domega0,
            omega2=omega2, omegamax=omegamax)

        pair = PairDensityCalculator(
            gs, context,
            threshold=threshold,
            nblocks=nblocks)
        assert len(extra_weights) == calc.numbands
        self.extra_weights_n = extra_weights
        self.extra_pair_weights_nm = self.set_extra_pair_weights()
        super().__init__(wd=wd, pair=pair, nbands=nbands, ecut=ecut, **kwargs)


    @timer('Get matrix element')
    def matrix_element(self, k_v, s):
        """Return pair density matrix element for integration.
        
        Overrides function in Chi0Calculator to add aditional
        weight in cRPA approach.
        """

        if self.optical:
            target_method = self.pair.get_optical_pair_density
            out_ngmax = self.qpd.ngmax + 2
        else:
            target_method = self.pair.get_pair_density
            out_ngmax = self.qpd.ngmax

        n_mnG = self._get_any_matrix_element(
            k_v, s, block=not self.optical,
            target_method=target_method,
        )
        n_nmG *= self.extra_pair_weights_nm[..., np.newaxis]**0.5

        
        return self.n_mnG.reshape(-1, out_ngmax)
