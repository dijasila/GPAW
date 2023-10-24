import numpy as np


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

        calculation = calc.calculation
        state = calculation.state
        self.ibzwfs = state.ibzwfs

        density = state.density
        self.collinear = density.collinear
        self.ndens = density.ndensities

        grid = density.nt_sR.desc
        self.ucvol = np.abs(np.linalg.det(grid.cell))
        self.bzvol = 2 * np.pi * np.abs(np.linalg.det(grid.icell))

        wfs = calc.wfs
        self.gd = wfs.gd
        self.nabla_aiiv = [setup.nabla_iiv for setup in calculation.setups]

        self.ns = None

    def get_plane_wave_coefficients(self, wfs, bands, spin):
        """
        Returns the periodic part of the real-space pseudo wfs
        for the given k-point, spin index and slice of band indices.

        Output is an array with shape (slice of band indices, 3D grid indices)
        """
        psit_nG = wfs.psit_nX[bands]
        G_plus_k_Gv = psit_nG.desc.G_plus_k_Gv
        return G_plus_k_Gv, self._pw_data(psit_nG, spin)

    def get_wave_function_projections(self, wfs, bands, spin):
        """
        Returns the projections of the pseudo wfs onto the partial waves
        for the given k-point, spin index and slice of band indices.

        Output is a dictionary with atom index keys and array values
        with shape (slice of band indices, partial wave index)
        """
        return self._proj_data(wfs.P_ani, bands, spin)

    def get_wfs(self, k_ind, spin):
        raise NotImplementedError

    def _pw_data(self, psit, spin):
        raise NotImplementedError

    def _proj_data(self, P, bands, spin):
        raise NotImplementedError


class CollinearGSInfo(GSInfo):
    def __init__(self, calc, comm):
        super().__init__(calc, comm)
        self.ns = self.ndens

    def get_wfs(self, k_ind, spin):
        return self.ibzwfs.wfs_qs[k_ind][spin]

    def _pw_data(self, psit_nG, _=None):
        return psit_nG.data

    def _proj_data(self, P_ani, bands, _=None):
        return {a: P_ni[bands] for a, P_ni in P_ani.items()}


class NoncollinearGSInfo(GSInfo):
    def __init__(self, calc, comm):
        super().__init__(calc, comm)
        self.ns = 2

    def get_wfs(self, k_ind, _=None):
        return self.ibzwfs.wfs_qs[k_ind][0]

    def _pw_data(self, psit_nsG, spin):
        return psit_nsG.data[:, spin]

    def _proj_data(self, P_ansi, bands, spin):
        return {a: P_nsi[bands, spin] for a, P_nsi in P_ansi.items()}
