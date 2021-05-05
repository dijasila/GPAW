import numpy as np
from ase import Atoms
from ase.parallel import parprint
from gpaw import GPAW, restart
from gpaw.elf import ELF
from gpaw.test import equal
from gpaw.mpi import rank, world


def test_utilities_elf(gpw_files):
    # Real wave functions
    atoms, calc = restart(gpw_files['h2_fd'])
    
    energy = atoms.get_potential_energy()
    elf = ELF(calc)
    elf.update()
    elf_G = elf.get_electronic_localization_function(gridrefinement=1)
    elf_g = elf.get_electronic_localization_function(gridrefinement=2)

    nt_G = calc.density.nt_sG[0]
    taut_G = elf.taut_sG[0]
    nt_grad2_G = elf.nt_grad2_sG[0]
    nt_grad2_g = elf.nt_grad2_sg[0]

    # integrate the H2 bond
    if rank == 0:
        # bond area
        x0 = atoms.positions[0][0] / atoms.get_cell()[0, 0]
        x1 = atoms.positions[1][0] / atoms.get_cell()[0, 0]
        y0 = (atoms.positions[0][1] - 1.0) / atoms.get_cell()[1, 1]
        y1 = 1 - y0
        z0 = (atoms.positions[0][2] - 1.0) / atoms.get_cell()[2, 2]
        z1 = 1 - z0
        gd = calc.wfs.gd
        Gx0, Gx1 = int(gd.N_c[0] * x0), int(gd.N_c[0] * x1)
        Gy0, Gy1 = int(gd.N_c[1] * y0), int(gd.N_c[1] * y1)
        Gz0, Gz1 = int(gd.N_c[2] * z0), int(gd.N_c[2] * z1)
        finegd = calc.density.finegd
        gx0, gx1 = int(finegd.N_c[0] * x0), int(finegd.N_c[0] * x1)
        gy0, gy1 = int(finegd.N_c[1] * y0), int(finegd.N_c[1] * y1)
        gz0, gz1 = int(finegd.N_c[2] * z0), int(finegd.N_c[2] * z1)
        int1 = elf_G[Gx0:Gx1, Gy0:Gy1, Gz0:Gz1].sum() * gd.dv
        int2 = elf_g[gx0:gx1, gy0:gy1, gz0:gz1].sum() * finegd.dv
        parprint("Ints", int1, int2)
        parprint("Min, max G", np.min(elf_G), np.max(elf_G))
        parprint("Min, max g", np.min(elf_g), np.max(elf_g))
    #   The tested values (< r7887) do not seem to be correct
        equal(int1, 14.579199, 0.0001)
        equal(int2, 18.936101, 0.0001)


    # Complex wave functions
    calc = GPAW(gpw_files['bcc_li_fd'])
    elf = ELF(calc)
    elf.update()
    elf_G = elf.get_electronic_localization_function(gridrefinement=1)
