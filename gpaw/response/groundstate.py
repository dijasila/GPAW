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
