
class GSInfo:
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

        wfs = calc.wfs
        self.gd = wfs.gd
        self.nabla_aiiv = [setup.nabla_iiv for setup in wfs.setups]

    def get_pseudo_wave_function(self, ni, nf, k_ind, spin):
        raise NotImplementedError

    def get_wave_function_projections(self, ni, nf, k_ind, spin):
        raise NotImplementedError


class CollinearGSInfo(GSInfo):
    def __init__(self, calc, comm):
        super().__init__(calc, comm)
        self.ns = self.ndens

    def get_pseudo_wave_function(self, ni, nf, k_ind, spin):
        psit_nX = self.ibzwfs.wfs_qs[k_ind][spin].psit_nX[ni:nf]
        return psit_nX.ifft(grid=self.grid, periodic=True).data

    def get_wave_function_projections(self, ni, nf, k_ind, spin):
        P_ani = {a: P_ni[ni:nf] for a, P_ni in
                 self.ibzwfs.wfs_qs[k_ind][spin].P_ani.items()}
        return P_ani


class NoncollinearGSInfo(GSInfo):
    def __init__(self, calc, comm):
        super().__init__(calc, comm)
        self.ns = 2

    def get_pseudo_wave_function(self, ni, nf, k_ind, spin):
        psit_nX = self.ibzwfs.wfs_qs[k_ind][0].psit_nX[ni:nf]
        return psit_nX.ifft(grid=self.grid, periodic=True).data[:, spin]

    def get_wave_function_projections(self, ni, nf, k_ind, spin):
        P_ani = {a: P_ni[ni:nf, spin] for a, P_ni in
                 self.ibzwfs.wfs_qs[k_ind][0].P_ani.items()}
        return P_ani
