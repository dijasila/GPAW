import numpy as np

from gpaw.symmetry import Symmetry


def reduce_kpoints(atoms, bzk_kc, setups, usesymm):
    """Reduce the number of k-points using symmetry.

    Returns symmetry object, weights and k-points in the irreducible
    part of the BZ."""

    if np.logical_and(np.logical_not(atoms.pbc), bzk_kc.any(axis=0)).any():
        raise ValueError('K-points can only be used with PBCs!')

    id_a = zip(atoms.get_initial_magnetic_moments(), setups.get_ids())

    # Construct a Symmetry instance containing the identity
    # operation only:
    symmetry = Symmetry(id_a, atoms.get_cell())

    if usesymm:
        # Find symmetry operations of atoms:
        symmetry.analyze(atoms.positions)

    # Reduce the set of k-points:
    ibzk_kc, weight_k = symmetry.reduce(bzk_kc)

    if usesymm:
        setups.set_symmetry(symmetry)
    else:
        symmetry = None

    return symmetry, weight_k, ibzk_kc
