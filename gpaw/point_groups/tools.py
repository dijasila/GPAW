import numpy as np
import math

def get_wfshape(symmetrycalc):
    """
    Get dimensions of a wave function. Assume all the functions are of the
    same size.
    """
    return symmetrycalc.get_wf(0).shape


def get_h(symmetrycalc):
    """
    Get grid spacings based on the cell dimensions
    and the wave function shape.
    """
    return (np.diagonal(symmetrycalc.atoms.get_cell()) /
            np.array(get_wfshape(symmetrycalc)))


def calculate_cutarea(atoms, coreatoms, wfshape, gridspacing, cutlimit=3.0):
    """
    Determine the volume where the symmetry analysis is constrained to.

    The constrained volume is within the `cutlimit` length units from each
    atom coordinate given in `coreatoms`.

    Parameters
    ----------
    atoms : ase.atoms
    coreatoms: array_like
        List of atom indices around which the symmetry analysis is run
    wfshape: array_like
        Shape of a wave function as np.ndarray(3)
    gridspacing : array_like
        grid spacings in length units as np.ndarray(3)
    cutlimit : float
        limit of cutting around an atom in length units
    """

    cutarea = np.zeros(wfshape)
    hx, hy, hz = gridspacing
    limit = cutlimit
    for c in coreatoms:
        x, y, z = np.array(atoms[c].position / gridspacing).astype(int)
        for i in range(x - int(limit/hx) - 1, x + int(limit/hx) + 1):
            for j in range(y - int(limit/hy) - 1, y + int(limit/hy) + 1):
                for k in range(z - int(limit/hz) - 1, z + int(limit/hz) + 1):
                    dist2 = ((i - x)**2 * hx**2 +
                             (j - y)**2 * hy**2 +
                             (k - z)**2 * hz**2)
                    if dist2 < cutlimit**2:
                        cutarea[i, j, k] = 1.
    return cutarea


def calculate_shiftvector(atoms, coreatoms, gridspacing):
    """
    Get the vector between the center of mass
    of `coreatoms` and the center of the computational cell.


    Parameters
    ----------
    atoms : ase.atoms
    coreatoms: array_like
        List of atom indices around which the analysis is run
    """

    # Get center of mass:
    cof = np.zeros(3)
    totalmass = 0.0
    for c in coreatoms:
        cof += atoms[c].mass * atoms[c].position
        totalmass += atoms[c].mass
    cof /= totalmass

    # Get cell center:
    cellcenter = np.diagonal(atoms.get_cell())/2.

    shiftvector = (cellcenter - cof) / gridspacing
    return shiftvector


def get_axis(atoms, indices):
    """
    Get a vector between two atoms.

    Parameters
    ----------
    atoms : ase.atoms
    indices : array_like
        The indices of the atoms of which the vector is solved
    """

    assert len(indices) == 2
    pos1 = atoms[indices[0]].position
    pos2 = atoms[indices[1]].position
    return pos1-pos2


def get_rotations_for_axes(mainaxis, secondaryaxis):
    """
    Solve the required rotations to have the main axis parallel to z
    and the secondary axis parallel to x.
    """

    # Main axis:
    x, y, z = mainaxis
    Rx = np.arctan2(y, z)
    Ry = np.arctan2(x, np.sqrt(z**2+y**2))

    # Rotate the secondary axis:
    rotmat1 = rotation_matrix([1, 0, 0], Rx)
    
    # minus sign is due to difference in rotation direction compared
    # to scipy.ndimage.interpolation.rotate:
    rotmat2 = rotation_matrix([0, 1, 0], -Ry)
    secondaryaxis = np.dot(rotmat1, secondaryaxis)
    secondaryaxis = np.dot(rotmat2, secondaryaxis)

    # Secondaryaxis:
    x, y, z = secondaryaxis
    Rz = -np.arctan2(y, x)

    rad2deg = 180./np.pi
    Rx *= rad2deg
    Ry *= rad2deg
    Rz *= rad2deg

    return Rx, Ry, Rz


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with
    counterclockwise rotation about the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
