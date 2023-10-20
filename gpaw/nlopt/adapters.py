

class GSInfo:
    """
    This is the base class for the ground state adapters in the non-linear
    optics module. It is only compatible with GPAW_NEW.

    The class should never be called directly, but should instead be called
    through the CollinearGSInfo and NoncollinearGSInfo classes.

    These subclasses are necessary due to the different ways which the spin
    index is handled in collinear and noncollinear ground state calculations.
    """
    def __init__(self, calc, comm):
        self.comm = comm  # This will eventually just become kptcomm from calc

        # We don't know why this is needed
        if calc.parameters.mode['name'] == 'lcao':
            calc.initialize_positions(calc.atoms)

        self.ibzk_kc = calc.get_ibz_k_points()
        self.nb_full = calc.get_number_of_bands()

        state = calc.calculation.state
        self.ibzwfs = state.ibzwfs

        density = state.density
        self.collinear = density.collinear
        self.grid = density.nt_sR.desc.new(dtype=complex)
        self.ndens = density.ndensities
        self.ns = None

        wfs = calc.wfs
        self.gd = wfs.gd
        self.nabla_aiiv = [setup.nabla_iiv for setup in wfs.setups]

    def get_pseudo_wave_function(self, ni, nf, k_ind, spin):
        """
        Returns the periodic part of the real-space pseudo wfs
        for the given k-point, spin index and slice of band indices.

        Output is an array with shape (slice of band indices, 3D grid indices)
        """
        wfs = self._get_wfs(k_ind, spin)
        psit_nX = wfs.psit_nX[ni:nf].ifft(grid=self.grid, periodic=True)
        return self._wfs_data(psit_nX, spin)

    def get_wave_function_projections(self, ni, nf, k_ind, spin):
        """
        Returns the projections of the pseudo wfs onto the partial waves
        for the given k-point, spin index and slice of band indices.

        Output is a dictionary with atom index keys and array values
        with shape (slice of band indices, partial wave index)
        """
        wfs = self._get_wfs(k_ind, spin)
        return self._proj_data(wfs.P_ani, ni, nf, spin)

    def _get_wfs(self, k_ind, spin):
        raise NotImplementedError

    def _wfs_data(self, psit, spin):
        raise NotImplementedError

    def _proj_data(self, P, ni, nf, spin):
        raise NotImplementedError


class CollinearGSInfo(GSInfo):
    def __init__(self, calc, comm):
        super().__init__(calc, comm)
        self.ns = self.ndens

    def _get_wfs(self, k_ind, spin):
        return self.ibzwfs.wfs_qs[k_ind][spin]

    def _wfs_data(self, psit_nR, _=None):
        return psit_nR.data

    def _proj_data(self, P_ani, ni, nf, _=None):
        return {a: P_ni[ni:nf] for a, P_ni in P_ani.items()}


class NoncollinearGSInfo(GSInfo):
    def __init__(self, calc, comm):
        super().__init__(calc, comm)
        self.ns = 2

    def _get_wfs(self, k_ind, _=None):
        return self.ibzwfs.wfs_qs[k_ind][0]

    def _wfs_data(self, psit_nsR, spin):
        return psit_nsR.data[:, spin]

    def _proj_data(self, P_ansi, ni, nf, spin):
        return {a: P_nsi[ni:nf, spin] for a, P_nsi in P_ansi.items()}
