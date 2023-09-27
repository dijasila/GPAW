import numpy as np

from ase.units import Bohr, Hartree
from ase.neighborlist import natural_cutoffs, build_neighbor_list

from gpaw.sphere.integrate import (integrate_lebedev,
                                   radial_truncation_function,
                                   spherical_truncation_function_collection,
                                   default_spherical_drcut,
                                   find_volume_conserving_lambd)
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.localft import (add_spin_polarization,
                                   add_LSDA_spin_splitting)


class AtomicSites:
    """Object defining a set of spherical atomic sites."""

    def __init__(self, indices, radii):
        """Construct the AtomicSites.

        Parameters
        ----------
        indices : 1D array-like
            Atomic index A for each site index a.
        radii : 2D array-like
            Atomic radius rc for each site index a and partitioning p.
        """
        self.A_a = np.asarray(indices)
        assert self.A_a.ndim == 1
        assert len(np.unique(self.A_a)) == len(self.A_a)

        # Parse the input atomic radii
        rc_ap = np.asarray(radii)
        assert rc_ap.ndim == 2
        assert rc_ap.shape[0] == len(self.A_a)
        # Convert radii to internal units (Å to Bohr)
        self.rc_ap = rc_ap / Bohr

        self.npartitions = self.rc_ap.shape[1]
        self.shape = rc_ap.shape

    def __len__(self):
        return len(self.A_a)


class AtomicSiteData:
    r"""Data object for a set of spherical atomic sites."""

    def __init__(self, gs: ResponseGroundStateAdapter, sites: AtomicSites):
        """Extract atomic site data from a ground state adapter."""
        assert self._in_valid_site_radii_range(gs, sites), \
            'Please provide site radii in the valid range, see '\
            'AtomicSiteData.valid_site_radii_range()'
        self.sites = sites

        # Extract the scaled positions and micro_setups for each atomic site
        self.spos_ac = gs.spos_ac[sites.A_a]
        self.micro_setup_a = [gs.micro_setups[A] for A in sites.A_a]

        # Extract pseudo density on the fine real-space grid
        self.finegd = gs.finegd
        self.nt_sr = gs.nt_sr

        # Set up the atomic truncation functions which define the sites based
        # on the coarse real-space grid
        self.gd = gs.gd
        self.drcut = default_spherical_drcut(self.gd)
        self.lambd_ap = np.array(
            [[find_volume_conserving_lambd(rcut, self.drcut)
              for rcut in rc_p] for rc_p in sites.rc_ap])
        self.stfc = spherical_truncation_function_collection(
            self.finegd, self.spos_ac, sites.rc_ap, self.drcut, self.lambd_ap)

    @staticmethod
    def _valid_site_radii_range(gs):
        """For each atom in gs, determine the valid site radii range in Bohr.

        The lower bound is determined by the spherical truncation width, when
        truncating integrals on the real-space grid.
        The upper bound is determined by the distance to the nearest
        augmentation sphere.
        """
        atoms = gs.atoms
        drcut = default_spherical_drcut(gs.gd)
        rmin_A = np.array([drcut / 2] * len(atoms))

        # Find neighbours based on covalent radii
        cutoffs = natural_cutoffs(atoms, mult=2)
        neighbourlist = build_neighbor_list(atoms, cutoffs,
                                            self_interaction=False)
        # Determine rmax for each atom
        augr_A = gs.get_aug_radii()
        rmax_A = []
        for A in range(len(atoms)):
            pos = atoms.positions[A]
            # Calculate the distance to the augmentation sphere of each
            # neighbour
            aug_distances = []
            for An, offset in zip(*neighbourlist.get_neighbors(A)):
                posn = atoms.positions[An] + offset @ atoms.get_cell()
                dist = np.linalg.norm(posn - pos) / Bohr  # Å -> Bohr
                aug_dist = dist - augr_A[An]
                assert aug_dist > 0.
                aug_distances.append(aug_dist)
            # In order for PAW corrections to be valid, we need a sphere of
            # radius rcut not to overlap with any neighbouring augmentation
            # spheres
            rmax_A.append(min(aug_distances))
        rmax_A = np.array(rmax_A)

        return rmin_A, rmax_A

    @staticmethod
    def valid_site_radii_range(gs):
        """Get the valid site radii for all atoms in a given ground state."""
        rmin_A, rmax_A = AtomicSiteData._valid_site_radii_range(gs)
        # Convert to external units (Bohr to Å)
        return rmin_A * Bohr, rmax_A * Bohr

    @staticmethod
    def _in_valid_site_radii_range(gs, sites):
        rmin_A, rmax_A = AtomicSiteData._valid_site_radii_range(gs)
        for a, A in enumerate(sites.A_a):
            if not np.all(
                    np.logical_and(
                        sites.rc_ap[a] > rmin_A[A] - 1e-8,
                        sites.rc_ap[a] < rmax_A[A] + 1e-8)):
                return False
        return True

    def calculate_magnetic_moments(self):
        """Calculate the magnetic moments at each atomic site."""
        magmom_ap = self.integrate_local_function(add_spin_polarization)
        return magmom_ap

    def calculate_spin_splitting(self):
        r"""Calculate the spin splitting Δ^(xc) for each atomic site."""
        dxc_ap = self.integrate_local_function(add_LSDA_spin_splitting)
        return dxc_ap * Hartree  # return the splitting in eV

    def integrate_local_function(self, add_f):
        r"""Integrate a local function f[n](r) = f(n(r)) over the atomic sites.

        For every site index a and partitioning p, the integral is defined via
        a smooth truncation function θ(|r-r_a|<rc_ap):

               /
        f_ap = | dr θ(|r-r_a|<rc_ap) f(n(r))
               /
        """
        out_ap = np.zeros(self.sites.shape, dtype=float)
        self._integrate_pseudo_contribution(add_f, out_ap)
        self._integrate_paw_correction(add_f, out_ap)
        return out_ap

    def _integrate_pseudo_contribution(self, add_f, out_ap):
        """Calculate the pseudo contribution to the atomic site integrals.

        For local functions of the density, the pseudo contribution is
        evaluated by a numerical integration on the real-space grid:

        ̰       /
        f_ap = | dr θ(|r-r_a|<rc_ap) f(ñ(r))
               /
        """
        # Evaluate the local function on the real-space grid
        ft_r = self.finegd.zeros()
        add_f(self.finegd, self.nt_sr, ft_r)

        # Integrate θ(|r-r_a|<rc_ap) f(ñ(r))
        ftdict_ap = {a: np.empty(self.sites.npartitions)
                     for a in range(len(self.sites))}
        self.stfc.integrate(ft_r, ftdict_ap)

        # Add pseudo contribution to the output array
        for a in range(len(self.sites)):
            out_ap[a] += ftdict_ap[a]

    def _integrate_paw_correction(self, add_f, out_ap):
        """Calculate the PAW correction to an atomic site integral.

        The PAW correction is evaluated on the atom centered radial grid, using
        the all-electron and pseudo densities generated from the partial waves:

                /
        Δf_ap = | r^2 dr θ(r<rc_ap) [f(n_a(r)) - f(ñ_a(r))]
                /
        """
        for a, (micro_setup, rc_p, lambd_p) in enumerate(zip(
                self.micro_setup_a, self.sites.rc_ap, self.lambd_ap)):
            # Evaluate the PAW correction and integrate angular components
            df_ng = micro_setup.evaluate_paw_correction(add_f)
            df_g = integrate_lebedev(df_ng)
            for p, (rcut, lambd) in enumerate(zip(rc_p, lambd_p)):
                # Evaluate the smooth truncation function
                theta_g = radial_truncation_function(
                    micro_setup.rgd.r_g, rcut, self.drcut, lambd)
                # Integrate θ(r) Δf(r) on the radial grid
                out_ap[a, p] += micro_setup.rgd.integrate_trapz(df_g * theta_g)