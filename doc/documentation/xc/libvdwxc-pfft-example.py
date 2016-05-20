from ase.structure import molecule
from gpaw import GPAW, PW
from gaw.xc.libvdwxc import vdw_df_cx

atoms = molecule('H2O')
calc = GPAW(mode=PW(500),
            xc=vdw_df_cx(mode='pfft', pfft_grid=(2, 2)),
            parallel=dict(augment_grids=True))
atoms.set_calculator(calc)
atoms.get_potential_energy()
