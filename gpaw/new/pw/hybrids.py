import numpy as np
from gpaw.core import PWArray, PWDesc, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.hybrids.paw import pawexxvv
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.utilities import unpack


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
        self.C_app = [setup.M_pp for setup in setups]

        self.v_G = coulomb(pw, grid, xc.exx_omega)
        self.ghat_aLG = setups.create_compensation_charges(
            pw, fracpos_ac, atomdist)
        self.plan = grid.fft_plans()

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit_nG: XArray,
                                spin: int,
                                Htpsit_nG: XArray) -> dict[str, float]:
        assert isinstance(psit_nG, PWArray)
        assert isinstance(Htpsit_nG, PWArray)
        energies = np.zeros(3)
        for wfs in ibzwfs:
            if wfs.spin != spin:
                continue
            D_aii = D_asii[:, spin]
            if ibzwfs.nspins == 1:
                D_aii.data *= 0.5
            energies += self.calculate(wfs, D_aii, psit_nG, Htpsit_nG)
        return {'evv': energies[0],
                'evc': energies[1],
                'ekin': energies[2]}

    def calculate(self,
                  wfs1: PWFDWaveFunctions,
                  D_aii,
                  psit2_nG: PWArray,
                  out_nG: PWArray):
        f1_n = wfs1.myocc_n
        psit1_nG = wfs1.psit_nX

        psit1_R = self.grid.empty()
        psit2_R = self.grid.empty()
        rhot_R = self.grid.empty()
        rhot_G = self.pw.empty()
        vrhot_G = self.pw.empty()

        P1_ani = wfs1.P_ani
        pt_aiG = wfs1.pt_aiX

        same = psit1_nG is psit2_nG
        if same:
            P2_ani = P1_ani
        else:
            P2_ani = pt_aiG.integrate(psit2_nG)

        evv = 0.0
        evc = 0.0
        ekin = 0.0
        B_ani = {}
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.C_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            B_ni = P2_ani[a] @ (-VC_ii - 2 * VV_ii)
            B_ani[a] = B_ni
            if same:
                ec = (D_ii * VC_ii).sum()
                ev = (D_ii * VC_ii).sum()
                ekin += ev + 2 * ev
                evv += ev
                evc += ec

        for n2, (psit2_G, out_G) in enumerate(zip(psit2_nG, out_nG)):
            psit2_G.ifft(out=psit2_R, plan=self.plan)

            for n1, (psit1_G, f1) in enumerate(zip(psit1_nG, f1_n)):
                psit1_G.ifft(out=psit1_R, plan=self.plan)
                rhot_R.data[:] = psit1_R.data * psit2_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                Q_aL = {a: np.einsum('i, ijL, j -> L',
                                     P1_ani[a][n1], delta_iiL, P2_ani[a][n2])
                        for a, delta_iiL in enumerate(self.delta_aiiL)}
                self.ghat_aLG.add_to(rhot_G, Q_aL)
                vrhot_G.data[:] = rhot_G.data * self.v_G.data
                if same:
                    e = f1 * f1_n[n2] * rhot_G.integrate(vrhot_G)
                    evv += 0.5 * e
                    ekin += e
                vrhot_G.ifft(out=rhot_R, plan=self.plan)
                rhot_R.data *= psit1_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                out_G.data -= rhot_G.data * f1

                A_aL = self.ghat_aLG.integrate(vrhot_G)
                for a, A_L in A_aL.items():
                    B_ani[a][n2] -= np.einsum(
                        'L, ijL, j -> i',
                        f1 * A_L, self.delta_aiiL[a], P1_ani[a][n1])

        pt_aiG.add_to(out_nG, B_ani)

        return evv, evc, ekin
