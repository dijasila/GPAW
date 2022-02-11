from __future__ import annotations
import numpy as np
from ase.units import Ha
from gpaw.typing import Array1D
from gpaw.new.calculation import DFTCalculation
from gpaw.new.gpw import write_gpw


def state(method):
    def new_method(self, *args, **kwargs):
        assert self.calculation is not None
        return method(self, self.calculation.state, *args, **kwargs)
    return new_method


class OldStuff:
    calculation: DFTCalculation | None

    def get_pseudo_wave_function(self, n):
        return self.calculation.ibzwfs[0].wave_functions.data[n]

    @state
    def get_fermi_level(self, state) -> float:
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        return fl[0]

    @state
    def get_homo_lumo(self, state, spin: int = None) -> Array1D:
        return state.ibzwfs.get_homo_lumo(spin) * Ha

    @state
    def get_number_of_electrons(self, state):
        return state.ibzwfs.nelectrons

    def get_atomic_electrostatic_potentials(self):
        _, _, Q_aL = self.calculation.pot_calc.calculate(
            self.calculation.state.density)
        Q_aL = Q_aL.gather()
        return Q_aL.data[::9] * (Ha / (4 * np.pi)**0.5)

    def write(self, filename, mode=''):
        """Write calculator object to a file.

        Parameters
        ----------
        filename
            File to be written
        mode
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        """
        self.log(f'Writing to {filename} (mode={mode!r})\n')

        write_gpw(filename, self.atoms, self.params,
                  self.calculation, skip_wfs=mode != 'all')
