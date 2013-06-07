import numpy as np
import ase.units as units

from gpaw.xc.hybrid import HybridXCBase


def forces(calc):
    """Return the atomic forces."""

    wfs = calc.wfs
    dens = calc.density
    ham = calc.hamiltonian

    assert not isinstance(ham.xc, HybridXCBase)

    calc.timer.start('Force calculation')

    natoms = len(wfs.setups)
    F_av = np.zeros((natoms, 3))

    # Force from projector functions (and basis set):
    wfs.calculate_forces(ham, F_av)

    try:
        # ODD functionals need force corrections for each spin
        correction = ham.xc.setup_force_corrections
    except AttributeError:
        pass
    else:
        correction(F_av)

    if wfs.bd.comm.rank == 0 and wfs.kd.comm.rank == 0:
        ham.calculate_forces(dens, F_av)

    wfs.world.broadcast(F_av, 0)

    F_av = wfs.symmetry.symmetrize_forces(F_av)

    calc.timer.stop('Force calculation')

    calc.text()
    calc.text('Forces in eV/Ang:')
    c = units.Hartree / units.Bohr
    symbols = calc.atoms.get_chemical_symbols()
    for a, symbol in enumerate(symbols):
        calc.text('%3d %-2s %10.5f %10.5f %10.5f' %
                  ((a, symbol) + tuple(F_av[a] * c)))

    return F_av
