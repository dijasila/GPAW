import numpy as np
from ase import Atoms
from gpaw.core import UGDesc
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new.ase_interface import GPAW
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


def vc_coulomb(wfs: PWFDWaveFunctions, grid: UGDesc):
    print(wfs.setups)
    pw = wfs.psit_nX.desc
    wstc = WignerSeitzTruncatedCoulomb(
        pw.cell_cv, np.array([1, 1, 1]))
    v_G = wstc.get_potential_new(pw, grid)
    ghat_aLG = wfs.setups.create_compensation_charges(
        pw, wfs.fracpos_ac, wfs.atomdist)
    nt_R = grid.empty()
    n = 0
    wfs.psit_nX[n].ifft(out=nt_R)
    nt_R.data *= nt_R.data
    rhot_G = pw.empty()
    nt_R.fft(out=rhot_G)
    P_ani = wfs.P_ani
    Q_aL = {a: np.einsum('i, ijL, j -> L',
                         P_ani[a][n], setup.Delta_iiL, P_ani[a][n])
            for a, setup in enumerate(wfs.setups)}
    ghat_aLG.add_to(rhot_G, Q_aL)
    rhot_G.data *= v_G.data
    e_aL = ghat_aLG.integrate(rhot_G)
    print(e_aL.data)


if __name__ == '__main__':
    atoms = Atoms('HLi')
    atoms.positions[1, 0] = 2.0
    atoms.center(vacuum=2.0)
    atoms.calc = GPAW(mode='pw')
    atoms.get_potential_energy()
    state = atoms.calc.dft.state
    vc_coulomb(state.ibzwfs.wfs_qs[0][0], state.density.nt_sR.desc)
