import numpy as np
from gpaw.core import PWDesc, UGDesc, PWArray
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.utilities import unpack
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


def coulomb(pw: PWDesc, grid: UGDesc, omega: float):
    if omega == 0.0:
        wstc = WignerSeitzTruncatedCoulomb(
            pw.cell_cv, np.array([1, 1, 1]))
        return wstc.get_potential_new(pw, grid)
    1 / 0
    """
        G2_G = pd.G2_qG[0]
        x_G = 1 - np.exp(-G2_G / (4 * self.omega**2))
        with np.errstate(invalid='ignore'):
            v_G = 4 * pi * x_G / G2_G
        G0 = G2_G.argmin()
        if G2_G[G0] < 1e-11:
            v_G[G0] = pi / self.omega**2
        return v_G
    """


class PWHybridHamiltonian(PWHamiltonian):
    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups,
                 fracpos_ac,
                 atomdist):
        super().__init__(grid, pw)
        self.grid = grid
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega
        self.VC_aii = [unpack(setup.X_p) for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.v_G = coulomb(pw, grid, xc.exx_omega)
        self.ghat_aLG = setups.create_compensation_charges(
            pw, fracpos_ac, atomdist)
        self.plan = grid.fft_plans()

    def apply_orbital_dependent(self,
                                ibzwfs,
                                psit_nG: XArray,
                                spin: int,
                                Htpsit_nG: XArray) -> float:
        assert isinstance(psit_nG, PWArray)
        assert isinstance(Htpsit_nG, PWArray)
        for wfs in ibzwfs:
            if wfs.spin != spin:
                continue
            self.calculate(wfs, psit_nG, Htpsit_nG)
        return 0.0

    def calculate(self,
                  wfs1: PWFDWaveFunctions,
                  psit2_nG: PWArray,
                  out_nG: PWArray):
        # same = wfs1.psit_nX is psit2_nG
        psit1_R = self.grid.empty()
        rhot_R = self.grid.empty()
        rhot_G = self.pw.empty()
        P1_ani = wfs1.P_ani
        P2_ani = wfs1.pt_aiX.integrate(psit2_nG)
        for n1, (psit1_G, f1) in enumerate(zip(wfs1.psit_nX, wfs1.myocc_n)):
            psit1_G.ifft(out=psit1_R, plan=self.plan)
            for n2, (psit2_G, out_G) in enumerate(zip(psit2_nG, out_nG)):
                psit2_G.ifft(out=rhot_R, plan=self.plan)
                rhot_R.data *= psit1_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                Q_aL = {a: np.einsum('i, ijL, j -> L',
                                     P1_ani[a][n1], delta_iiL, P2_ani[a][n2])
                        for a, delta_iiL in enumerate(self.delta_aiiL)}
                self.ghat_aLG.add_to(rhot_G, Q_aL)
                rhot_G.data *= self.v_G.data
                rhot_G.ifft(out=rhot_R, plan=self.plan)
                rhot_R.data *= psit1_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                out_G.data -= rhot_G.data * f1
