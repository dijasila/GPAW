import numpy as np

from gpaw.mpi import MPIComm
from gpaw.new.ase_interface import ASECalculator


class GSInfo:
    """
    This is the base class for the ground state adapters in the non-linear
    optics module. It is only compatible with GPAW_NEW.

    The class should never be called directly, but should instead be called
    through the CollinearGSInfo or NoncollinearGSInfo classes.

    These subclasses are necessary due to the different ways which the spin
    index is handled in collinear and noncollinear ground state calculations.
    """
    def __init__(self,
                 calc: ASECalculator,
                 comm: MPIComm):
        self.comm = comm  # This will eventually just become kptcomm from calc
        assert calc.params.mode['name'] == 'pw', \
            'Calculator must be in plane wave mode.'

        calculation = calc.calculation
        self.nabla_aiiv = [setup.nabla_iiv for setup in calculation.setups]

        state = calculation.state
        ibzwfs = state.ibzwfs
        self.ibzwfs = ibzwfs
        assert ((ibzwfs.band_comm.size == 1) and
                (ibzwfs.domain_comm.size == 1) and
                (ibzwfs.kpt_comm.size == 1) and
                (ibzwfs.kpt_band_comm.size == 1)), \
            ('You must initialise the calculator '
             'with "communicator = serial_comm".')

        density = state.density
        self.collinear = density.collinear
        self.ndensities = density.ndensities

        grid = density.nt_sR.desc
        self.ucvol = np.abs(np.linalg.det(grid.cell))
        self.bzvol = np.abs(np.linalg.det(2 * np.pi * grid.icell))

        self.ns: int

    def get_plane_wave_coefficients(self, wfs, bands, spin):
        """
        Returns the plane wave coefficients and reciprocal vectors.

        Output is an array with shape (band index, reciprocal vector index)
        """
        psit_nG = wfs.psit_nX[bands]
        G_plus_k_Gv = psit_nG.desc.G_plus_k_Gv
        return G_plus_k_Gv, self._pw_data(psit_nG, spin)

    def get_wave_function_projections(self, wfs, bands, spin):
        """
        Returns the projections of the pseudo wfs onto the partial waves.

        Output is a dictionary with atom index keys and array values with
        shape (band index, partial wave index)
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
        self.ns = self.ndensities

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
