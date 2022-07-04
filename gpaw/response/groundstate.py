import numpy as np
from ase.units import Ha


class ResponseGroundStateAdapter:
    def __init__(self, calc):
        wfs = calc.wfs

        self.kd = wfs.kd
        self.world = calc.world
        self.gd = wfs.gd
        self.bd = wfs.bd
        self.pd = wfs.pd
        self.nspins = wfs.nspins
        self.dtype = wfs.dtype

        self.spos_ac = calc.spos_ac

        self.wfs = wfs
        self.kpt_u = wfs.kpt_u
        self.kpt_qs = wfs.kpt_qs
        self.setups = wfs.setups

        self.fermi_level = wfs.fermi_level
        self.atoms = calc.atoms
        self.pbc = self.atoms.pbc
        self.volume = self.gd.volume

        self.nvalence = wfs.nvalence
        self._density = calc.density
        self._hamiltonian = calc.hamiltonian
        self._calc = calc

    def get_occupations_width(self):
        # Ugly hack only used by pair.intraband_pair_density I think.
        # Actually: was copy-pasted in chi0 also.
        # More duplication can probably be eliminated around those.

        # Only works with Fermi-Dirac distribution
        occs = self.wfs.occupations
        assert occs.name in {'fermi-dirac', 'zero-width'}

        # No carriers when T=0
        width = getattr(occs, '_width', 0.0) / Ha
        return width

    def nonpbc_cell_product(self):
        """Volume, area, or length, taken in all non-periodic directions."""
        nonpbc = ~self.pbc
        cell_cv = self.gd.cell_cv
        return abs(np.linalg.det(cell_cv[nonpbc][:, nonpbc]))

    @property
    def nt_sG(self):
        # Used by kxc
        return self._density.nt_sG

    @property
    def D_asp(self):
        # Used by kxc
        return self._density.D_asp

    def all_electron_density(self, gridrefinement=2):
        # used by kxc
        return self._density.get_all_electron_density(
            atoms=self.atoms, gridrefinement=gridrefinement)

    # Things used by EXX.  This is getting pretty involved.
    #
    # EXX naughtily accesses the density object in order to
    # interpolate_pseudo_density() which is in principle mutable.

    # Used by EXX.
    @property
    def hamiltonian(self):
        return self._hamiltonian

    # Used by EXX.
    @property
    def density(self):
        return self._density

    # Ugh SOC
    def soc_eigenstates(self, **kwargs):
        from gpaw.spinorbit import soc_eigenstates
        return soc_eigenstates(self._calc, **kwargs)

    @property
    def xcname(self):
        return self.hamiltonian.xc.name

    # XXX This is used by xc == JGMsx from g0w0
    def get_band_gap(self):
        from ase.dft.bandgap import get_band_gap
        gap, k1, k2 = get_band_gap(self._calc)
        return gap
