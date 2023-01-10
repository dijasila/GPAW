from pathlib import Path

import numpy as np

from ase.units import Ha, Bohr

import gpaw.mpi as mpi


class ResponseGroundStateAdapter:
    def __init__(self, calc):
        wfs = calc.wfs

        self.kd = wfs.kd
        self.world = calc.world
        self.gd = wfs.gd
        self.finegd = calc.density.finegd
        self.bd = wfs.bd
        self.nspins = wfs.nspins
        self.dtype = wfs.dtype

        self.spos_ac = calc.spos_ac

        self.kpt_u = wfs.kpt_u
        self.kpt_qs = wfs.kpt_qs
        self.setups = wfs.setups

        self.fermi_level = wfs.fermi_level
        self.atoms = calc.atoms
        self.pbc = self.atoms.pbc
        self.volume = self.gd.volume

        self.nvalence = wfs.nvalence

        self._wfs = wfs
        self._density = calc.density
        self._hamiltonian = calc.hamiltonian
        self._calc = calc

    @staticmethod
    def from_gpw_file(gpw, context=None):
        """Initiate the ground state adapter directly from a .gpw file."""
        from gpaw import GPAW, disable_dry_run
        assert Path(gpw).is_file()

        if context is None:
            def timer(*unused):
                def __enter__(self):
                    pass

                def __exit__(self):
                    pass
        else:
            timer = context.timer
            context.print('Reading ground state calculation:\n  %s' % gpw)

        with timer('Read ground state'):
            with disable_dry_run():
                calc = GPAW(gpw, txt=None, communicator=mpi.serial_comm)

        return ResponseGroundStateAdapter(calc)

    @property
    def pd(self):
        # This is an attribute error in FD/LCAO mode.
        # We need to abstract away "calc" in all places used by response
        # code, and that includes places that are also compatible with FD.
        return self._wfs.pd

    def get_occupations_width(self):
        # Ugly hack only used by pair.intraband_pair_density I think.
        # Actually: was copy-pasted in chi0 also.
        # More duplication can probably be eliminated around those.

        # Only works with Fermi-Dirac distribution
        occs = self._wfs.occupations
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
    def nt_sR(self):
        # Used by fxc_kernels
        return self._density.nt_sG

    @property
    def D_asp(self):
        # Used by fxc_kernels
        return self._density.D_asp

    def pseudo_density(self, gridrefinement=2):
        # Used by localft
        if gridrefinement == 1:
            return self.nt_sR, self.gd
        elif gridrefinement == 2:
            nt_sr = self._density.interpolate_pseudo_density()
            return nt_sr, self.finegd
        else:
            raise ValueError(f'Invalid gridrefinement {gridrefinement}')

    def all_electron_density(self, gridrefinement=2):
        # Used by fxc_kernels and localft
        return self._density.get_all_electron_density(
            atoms=self.atoms, gridrefinement=gridrefinement)

    # Things used by EXX.  This is getting pretty involved.
    #
    # EXX naughtily accesses the density object in order to
    # interpolate_pseudo_density() which is in principle mutable.

    def hacky_all_electron_density(self, **kwargs):
        # fxc likes to get all electron densities.  It calls
        # calc.get_all_electron_density() and so we wrap that here.
        # But it also collects to serial (bad), and it also zeropads
        # nonperiodic directions (probably WRONG!).
        #
        # Also this one returns in user units, whereas the calling
        # code actually wants internal units.  Very silly then.
        #
        # ALso, the calling code often wants the gd, which is not
        # returned, so it is redundantly reconstructed in multiple
        # places by refining the "right" number of times.
        n_g = self._calc.get_all_electron_density(**kwargs)
        n_g *= Bohr**3
        return n_g

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

    def get_xc_difference(self, xc):
        # XXX used by gpaw/xc/tools.py
        return self._calc.get_xc_difference(xc)

    def get_wave_function_array(self, u, n):
        # XXX used by gpaw/xc/tools.py in a hacky way
        return self._wfs._get_wave_function_array(
            u, n, realspace=True)

    def pair_density_paw_corrections(self, pd):
        from gpaw.response.paw import get_pair_density_paw_corrections
        return get_pair_density_paw_corrections(
            setups=self.setups, pd=pd, spos_ac=self.spos_ac)

    def get_pos_av(self):
        # gd.cell_cv must always be the same as pd.gd.cell_cv, right??
        return np.dot(self.spos_ac, self.gd.cell_cv)

    def count_occupied_bands(self, ftol):
        nocc1 = 9999999
        nocc2 = 0
        for kpt in self.kpt_u:
            f_n = kpt.f_n / kpt.weight
            nocc1 = min((f_n > 1 - ftol).sum(), nocc1)
            nocc2 = max((f_n > ftol).sum(), nocc2)
        return nocc1, nocc2

    @property
    def ibzq_qc(self):
        # For G0W0Kernel
        kd = self.kd
        bzq_qc = kd.get_bz_q_points(first=True)
        U_scc = kd.symmetry.op_scc
        ibzq_qc = kd.get_ibz_q_points(bzq_qc, U_scc)[0]

        return ibzq_qc
