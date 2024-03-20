from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.core.arrays import DistributedArrays as XArray


class PWHybridHamiltonian(PWHamiltonian):
    def __init__(self, grid, pw, xc, VC_aii, delta_aiiL):
        super().__init__(grid, pw)
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega
        self.VC_aii = VC_aii
        self.delta_aiiL = delta_aiiL
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)
        v_G = coulomb.get_potential(pd12)

    def apply_orbital_dependent(self,
                                ibzwfs,
                                psit_nG: XArray,
                                spin: int,
                                Htpsit_nG: XArray) -> float:
        for wfs in ibzwfs:
            if wfs.spin != spin:
                continue
            print(wfs.psit_nX)
        return 0.0
