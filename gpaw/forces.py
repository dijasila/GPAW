import numpy as np

from gpaw.xc.hybrid import HybridXCBase


class ForceCalculator:
    def __init__(self, timer):
        self.timer = timer
        self.reset()
        
    def reset(self):
        self.F_av = None

    def calculate(self, wfs, dens, ham):
        """Return the atomic forces."""

        assert not isinstance(ham.xc, HybridXCBase)

        if self.F_av is not None:
            return self.F_av

        self.timer.start('Force calculation')

        natoms = len(wfs.setups)

        # Force from projector functions (and basis set):
        F_wfs_av = np.zeros((natoms, 3))
        wfs.calculate_forces(ham, F_wfs_av)
        wfs.gd.comm.sum(F_wfs_av, 0)
                
        F_ham_av = np.zeros((natoms, 3))

        try:
            # ODD functionals need force corrections for each spin
            correction = ham.xc.setup_force_corrections
        except AttributeError:
            pass
        else:
            correction(F_ham_av)

        ham.calculate_forces(dens, F_ham_av)

        F_av = F_ham_av + F_wfs_av
        wfs.world.broadcast(F_av, 0)

        self.F_av = wfs.kd.symmetry.symmetrize_forces(F_av)
        self.timer.stop('Force calculation')
        return self.F_av
