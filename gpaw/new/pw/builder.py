import _gpaw
import numpy as np
from ase.units import Ha
from gpaw.core import PlaneWaves
from gpaw.core.plane_waves import PlaneWaveExpansions
from gpaw.new.builder import create_uniform_grid
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pw.poisson import ReciprocalSpacePoissonSolver
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from gpaw.new.pwfd.builder import PWFDDFTComponentsBuilder


class PWDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    interpolation = 'fft'

    def __init__(self, atoms, params, ecut=340):
        self.ecut = ecut / Ha
        super().__init__(atoms, params)

        self._nct_ag = None

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            'pw',
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation='fft',
            ecut=self.ecut,
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> PlaneWaves:
        assert self.grid.comm.size == 1
        return PlaneWaves(ecut=self.ecut,
                          cell=self.grid.cell,
                          dtype=self.dtype)

    def create_xc_functional(self):
        if self.params.xc['name'] in ['HSE06', 'PBE0', 'EXX']:
            return ...
        return super().create_xc_functional()

    def get_pseudo_core_densities(self):
        if self._nct_ag is None:
            pw = PlaneWaves(ecut=2 * self.ecut, cell=self.grid.cell)
            self._nct_ag = self.setups.create_pseudo_core_densities(
                pw, self.fracpos_ac)
        return self._nct_ag

    def create_poisson_solver(self, fine_grid_pw, params):
        return ReciprocalSpacePoissonSolver(fine_grid_pw)

    def create_potential_calculator(self):
        nct_ag = self.get_pseudo_core_densities()
        pw = nct_ag.pw
        fine_pw = pw.new(ecut=8 * self.ecut)
        poisson_solver = self.create_poisson_solver(
            fine_pw,
            self.params.poissonsolver)
        return PlaneWavePotentialCalculator(self.grid,
                                            self.fine_grid,
                                            pw,
                                            fine_pw,
                                            self.setups,
                                            self.xc,
                                            poisson_solver,
                                            nct_ag, self.nct_R)

    def create_hamiltonian_operator(self, blocksize=10):
        return PWHamiltonian()

    def convert_wave_functions_from_uniform_grid(self, psit_nR):
        pw = self.wf_desc.new(kpt=psit_nR.desc.kpt_c)
        psit_nG = pw.empty(psit_nR.dims, psit_nR.comm)
        for psit_R, psit_G in zip(psit_nR, psit_nG):
            psit_R.fft(out=psit_G)
        return psit_nG

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'coefficients' not in reader.wave_functions:
            return ibzwfs

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file
        elif reader.version < 4:
            c /= self.grid.size_c.prod()

        index_kG = reader.wave_functions.indices

        for wfs in ibzwfs:
            pw = self.wf_desc.new(kpt=wfs.kpt_c)
            if wfs.spin == 0:
                index_G = pw.indices(tuple(self.grid.size))
                nG = len(index_G)
                assert (index_G == index_kG[wfs.k, :nG]).all()
                assert (index_kG[wfs.k, nG:] == -1).all()

            index = (wfs.spin, wfs.k) if self.ncomponents != 4 else (wfs.k,)
            data = reader.wave_functions.proxy('coefficients', *index)
            data.scale = c
            data.length_of_last_dimension = pw.shape[0]
            orig_shape = data.shape
            data.shape = (self.nbands, ) + pw.shape
            wfs.psit_nX = PlaneWaveExpansions(pw, self.nbands,
                                              data=data)
            data.shape = orig_shape

        return ibzwfs


class PWHamiltonian(Hamiltonian):
    def apply(self, vt_sR, psit_nG, out, spin):
        out_nG = out
        vt_R = vt_sR.data[spin]
        np.multiply(psit_nG.desc.ekin_G, psit_nG.data, out_nG.data)
        grid = vt_sR.desc
        if psit_nG.desc.dtype == complex:
            grid = grid.new(dtype=complex)
        f_R = grid.empty()
        for p_G, o_G in zip(psit_nG, out_nG):
            f_R = p_G.ifft(out=f_R)
            f_R.data *= vt_R
            o_G.data += f_R.fft(pw=p_G.desc).data
        return out_nG

    def create_preconditioner(self, blocksize):
        return precondition


def precondition(psit, residuals, out):
    G2 = psit.desc.ekin_G * 2
    for r, o, ekin in zip(residuals.data, out.data, psit.norm2('kinetic')):
        _gpaw.pw_precond(G2, r, ekin, o)
