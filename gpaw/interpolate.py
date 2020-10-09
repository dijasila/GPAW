from ase import Atoms
from ase.units import Bohr

from . import GPAW
from .wavefuntions.pw import PWWaweFunctions

# check cell symm


def interpolate(calc: GPAW,
                atoms: Atoms):
    wfs = calc.wfs

    N_c, h = calc.choose_number_of_grid_points()
    gd = wfs.gd.new_descriptor(N_c=N_c, cell_cv=atoms.cell / Bohr)

    calc.wfs = PWWaweFunctions(
        wfs.ecut,
        wfs.gammacentered,
        wfs.fftwflags,
        wfs.dedepsilon,
        (calc.comms['K'],) + wfs.scalapack_parameters[1:],
        wfs.initksl,
        wfs.wfs_mover,
        wfs.collinear,
        gd,
        wfs.nvalence,
        wfs.setups,
        wfs.bd,
        wfs.dtype,
        wfs.world,
        wfs.kd,
        wfs.kptband_comm,
        wfs.timer)

    calc.create_density(False, 'pw', dens.background_charge, h)
    