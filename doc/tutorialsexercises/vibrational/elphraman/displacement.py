from ase.io import read
from gpaw import GPAW, FermiDirac, PW
from gpaw.elph import DisplacementRunner


def get_pw_calc(ecut, kpts):
    return GPAW(mode=PW(ecut),
                parallel={'sl_auto': True, 'augment_grids': True},
                xc='PBE',
                kpts=kpts,
                occupations=FermiDirac(0.02),
                convergence={'energy': 1e-6,
                             'density': 1e-6},
                symmetry={'point_group': False},
                txt='displacement.txt')


atoms = read("MoS2_2H_relaxed_PBE.json")
calc = get_pw_calc(700, (2, 2, 2))
elph = DisplacementRunner(atoms=atoms, calc=calc,
                          supercell=(3, 3, 2), name='elph',
                          calculate_forces=True)
elph.run()
