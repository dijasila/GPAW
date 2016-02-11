import numpy as np
from ase.units import Bohr
from gpaw.grid_descriptor import GridDescriptor


def extended_grid_descriptor(gd,
                             extend_N_cd=None,
                             N_c=None, extcomm=None):
    """Create grid descriptor for extended grid.

    Provide only either extend_N_cd or N_c.

    Parameters:

    extend_N_cd: ndarray, int
        Number of extra grid points per axis (c) and
        direction (d, left or right)
    N_c: ndarray, int
        Number of grid points in extended grid
    extcomm:
        Communicator for the extended grid, defaults to gd.comm
    """

    if extcomm is None:
        extcomm = gd.comm

    if extend_N_cd is None:
        assert N_c is not None, 'give only extend_N_cd or N_c'
        N_c = np.array(N_c, dtype=np.int)
        extend_N_cd = np.tile((N_c - gd.N_c) // 2, (2, 1)).T
    else:  # extend_N_cd is not None:
        assert N_c is None, 'give only extend_N_cd or N_c'
        extend_N_cd = np.array(extend_N_cd, dtype=np.int)
        N_c = gd.N_c + extend_N_cd.sum(axis=1)

    cell_cv = gd.h_cv * N_c
    move_c = gd.get_grid_spacings() * extend_N_cd[:, 0]

    egd = GridDescriptor(N_c, cell_cv, gd.pbc_c, extcomm)
    egd.extend_N_cd = extend_N_cd

    return egd, cell_cv * Bohr, move_c * Bohr


def extend_array(d_g, gd, d_e, egd):
    big_d_e = egd.collect(d_e)
    big_d_g = gd.collect(d_g)

    N_cd = egd.extend_N_cd

    if egd.comm.rank == 0:
        assert gd.comm.rank == 0, \
            'extended array master has to equal to the grid descriptor master'
        N1_c = N_cd[:, 0]
        N2_c = N1_c + gd.N_c - 1  # implicit zero
        big_d_e[N1_c[0]:N2_c[0], N1_c[1]:N2_c[1], N1_c[2]:N2_c[2]] = big_d_g
    egd.distribute(big_d_e, d_e)


def deextend_array(d_g, gd, d_e, egd):
    big_d_g = gd.collect(d_g)
    big_d_e = egd.collect(d_e)

    N_cd = egd.extend_N_cd

    if egd.comm.rank == 0:
        assert gd.comm.rank == 0, \
            'extended array master has to equal to the grid descriptor master'
        N1_c = N_cd[:, 0]
        N2_c = N1_c + gd.N_c - 1  # implicit zero
        big_d_g[:] = big_d_e[N1_c[0]:N2_c[0], N1_c[1]:N2_c[1], N1_c[2]:N2_c[2]]
    elif gd.comm.rank == 0:
        # Fill with nan to find out errors with different communicators
        big_d_g[:] = np.nan
        # TODO:
        # If you have used for egd other communicator
        # than gd.comm, you have to call appropriate
        # broadcast to distribute d_g to all cpus.

    gd.distribute(big_d_g, d_g)


def move_atoms(atoms, move_c):
    pos_a = atoms.get_positions()
    for pos in pos_a:
        pos += move_c
    atoms.set_positions(pos_a)
