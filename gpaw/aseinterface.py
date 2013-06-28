# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

import os

import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator, ReadError, Parameters

import gpaw.io
import gpaw.mpi as mpi
from gpaw.xc import XC
from gpaw.paw import PAW
from gpaw.hooks import hooks
from gpaw.stress import stress
from gpaw.forces import forces
from gpaw.parameters import read_parameters
from gpaw.occupations import MethfesselPaxton
from gpaw.wavefunctions.base import EmptyWaveFunctions
from gpaw.wavefunctions.pw import ReciprocalSpaceDensity
from gpaw import parsize_domain, parsize_bands, sl_default, sl_diagonalize, \
                 sl_inverse_cholesky, sl_lcao, buffer_size, \
                 KohnShamConvergenceError


class GPAW(PAW, Calculator):
    """This is the ASE-calculator frontend for doing a PAW calculation."""

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms']

    default_parameters = {
        'h':               None,  # Angstrom
        'xc':              'LDA',
        'gpts':            None,
        'kpts':            [(0, 0, 0)],
        'lmax':            2,
        'charge':          0,
        'nbands':          None,
        'setups':          'paw',
        'basis':           None,
        'smearing':        None,
        'occupations':     None,
        'spinpol':         None,
        'usesymm':         True,
        'stencils':        (3, 3),
        'fixdensity':      False,
        'mixer':           None,
        'hund':            False,
        'random':          False,
        'dtype':           None,
        'maxiter':         120,
        'external':        None,  # eV
        'eigensolver':     None,
        'poissonsolver':   None,
        'idiotproof':      True,
        'mode':            'fd',
        'realspace':       None,
        'filter':          None,
        'convergence':     {'energy':      0.0005,  # eV / electron
                            'density':     1.0e-4,
                            'eigenstates': 4.0e-8,  # eV^2
                            'bands':       'occupied'},
        }

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None,
                 txt='__default__', verbose=0, 
                 communicator=None, parallel=None,
                 **kwargs):

        self.txt = None
        self.world = mpi.world
        self.parallel = {'kpt':                 None,
                         'domain':              parsize_domain,
                         'band':                parsize_bands,
                         'stridebands':         False,
                         'sl_auto':             False,
                         'sl_default':          sl_default,
                         'sl_diagonalize':      sl_diagonalize,
                         'sl_inverse_cholesky': sl_inverse_cholesky,
                         'sl_lcao':             sl_lcao,
                         'buffer_size':         buffer_size}

        if label is not None and not label.endswith('.gpw'):
            label += '.gpw'
        if restart is not None and not restart.endswith('.gpw'):
            restart += '.gpw'

        if communicator is not None:
            self.set_communicator(communicator)
        if parallel is not None:
            self.set_parallel(parallel)
        if txt == '__default__':
            if label is None:
                txt = '-'
            else:
                txt = label[:-3] + 'txt'
        self.set_text(txt, verbose)

        PAW.__init__(self)

        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)
        self.niterations = None

    def read(self, label):
        self.set_label(label)
        if self.label is None or not os.path.isfile(self.label):
            raise ReadError

        reader = gpaw.io.open(self.label, 'r', self.world)
        self.atoms = gpaw.io.read_atoms(reader)
        self.parameters = self.get_default_parameters()
        read_parameters(self.parameters, reader)
        self.initialize()
        gpaw.io.read(self, reader)
        self.print_cell_and_parameters()

    def set_communicator(self, communicator):
        if isinstance(communicator, (list, np.ndarray)):
            self.world = mpi.world.new_communicator(np.asarray(communicator))
        else:
            self.world = communicator

    def set_parallel(self, parallel):
        for key in parallel:
            assert key in self.parallel
        self.parallel.update(parallel)

    def reset(self):
        if self.scf is not None:
            self.scf.reset()
        Calculator.reset(self)

    def set(self, **kwargs):
        if 'txt' in kwargs:
            self.set_text(kwargs.pop('txt'), self.verbose)

        if (kwargs.get('h') is not None) and (kwargs.get('gpts') is not None):
            raise TypeError("""You can't use both "gpts" and "h"!""")

        changed_parameters = Calculator.set(self, **kwargs)

        p = self.parameters

        for key in changed_parameters:
            self.initialized = False
            
            if key == 'basis' and p['mode'] == 'fd':
                continue

            if key == 'eigensolver':
                self.wfs.set_eigensolver(None)
            
            if key in ['mixer', 'verbose', 'hund', 'random',
                       'eigensolver', 'idiotproof']:
                continue

            self.results = {}

            if key in ['convergence', 'fixdensity', 'maxiter']:
                self.scf = None
                continue

            # More drastic changes:
            self.scf = None
            self.wfs.set_orthonormalized(False)
            if key in ['lmax', 'stencils', 'external', 'xc',
                       'poissonsolver', 'occupations', 'smearing']:
                self.hamiltonian = None
                self.occupations = None
            elif key in ['charge']:
                self.hamiltonian = None
                self.density = None
                self.wfs = EmptyWaveFunctions()
                self.occupations = None
            elif key in ['kpts', 'nbands', 'usesymm']:
                self.wfs = EmptyWaveFunctions()
                self.occupations = None
            elif key in ['h', 'gpts', 'setups', 'spinpol', 'realspace',
                         'dtype', 'mode']:
                self.density = None
                self.occupations = None
                self.hamiltonian = None
                self.wfs = EmptyWaveFunctions()
            elif key in ['basis']:
                self.wfs = EmptyWaveFunctions()
            elif key in ['parsize', 'parsize_bands', 'parstride_bands']:
                name = {'parsize': 'domain',
                        'parsize_bands': 'band',
                        'parstride_bands': 'stridebands'}[key]
                raise DeprecationWarning(
                    'Keyword argument has been moved ' +
                    "to the 'parallel' dictionary keyword under '%s'." % name)
            else:
                raise TypeError("Unknown keyword argument: '%s'" % key)

    def get_stress(self, atoms=None):
        if self.parameters.mode in ['fd', 'lcao']:
            raise NotImplementedError
        return Calculator.get_stress(self, atoms)

    def get_dipole_moment(self, atoms=None):
        if isinstance(self.density, ReciprocalSpaceDensity):
            raise NotImplementedError
        return Calculator.get_dipole_moment(self, atoms)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges','magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        for change in ['numbers', 'pbc', 'cell', 'magmoms']:
            if change in system_changes:
                self.wfs = EmptyWaveFunctions()
                self.occupations = None
                self.density = None
                self.hamiltonian = None
                self.scf = None
                self.initialize()
                self.print_unit_cell()
                self.set_positions()
                break
        else:
            if 'positions' in system_changes:
                if self.density is not None:
                    self.density.reset()
                self.set_positions()

        if not self.initialized:
            self.initialize()
            self.set_positions()
            
        converged = False
        if not self.scf.converged:
            self.timer.start('SCF-cycle')
            for niterations in self.scf.run(self.wfs, self.hamiltonian,
                                            self.density, self.occupations):
                self.call_observers(niterations)
                self.print_iteration(niterations)
                self.niterations = niterations
            self.timer.stop('SCF-cycle')

            if self.scf.converged:
                converged = True
                self.print_converged(niterations)
            else:
                if 'not_converged' in hooks:
                    hooks['not_converged'](self)
                raise KohnShamConvergenceError('Did not converge!')

        self.results['free_energy'] = Hartree * self.hamiltonian.Etot
        self.results['energy'] = Hartree * (self.hamiltonian.Etot +
                                            0.5 * self.hamiltonian.S)

        if 'forces' in properties or 'stress' in properties:
            self.converge_wave_functions()
            if not self.wfs.positions_set:
                self.set_positions()

        if 'forces' in properties:
            F_av = forces(self)
            self.results['forces'] = F_av * (Hartree / Bohr)

        if 'stress' in properties:
            stress_vv = stress(self)
            self.results['stress'] = (stress_vv.flat[[0, 4, 8, 5, 2, 1]] *
                                      (Hartree / Bohr**3))

        if 'dipole' in properties:
            dipole_v = self.calculate_dipole_moment()
            self.results['dipole'] =  dipole_v * Bohr
        
        if self.wfs.nspins == 2 or not self.wfs.collinear:
            magmom, magmoms = self.calculate_magnetic_moments()
            self.results['magmom'] = magmom
            self.results['magmoms'] = magmoms

        if self.label:
            self.write(self.label)

        if converged:
            self.call_observers(niterations, final=True)
            if 'converged' in hooks:
                hooks['converged'](self)

    def get_number_of_bands(self):
        """Return the number of bands."""
        return self.wfs.bd.nbands
  
    def get_xc_functional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.hamiltonian.xc.name
 
    def get_number_of_spins(self):
        return self.wfs.nspins

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        return self.wfs.nspins == 2
    
    def get_bz_k_points(self):
        """Return the k-points."""
        return self.wfs.kd.bzk_kc.copy()
 
    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.wfs.kd.ibzk_kc.copy()

    def get_k_point_weights(self):
        """Weights of the k-points.
        
        The sum of all weights is one."""
        
        return self.wfs.weight_k

    def get_pseudo_density(self, spin=None, gridrefinement=1,
                           pad=True, broadcast=True):
        """Return pseudo-density array.

        If *spin* is not given, then the total density is returned.
        Otherwise, the spin up or down density is returned (spin=0 or
        1)."""

        if gridrefinement == 1:
            nt_sG = self.density.nt_sG
            gd = self.density.gd
        elif gridrefinement == 2:
            if self.density.nt_sg is None:
                self.density.interpolate_pseudo_density()
            nt_sG = self.density.nt_sg
            gd = self.density.finegd
        else:
            raise NotImplementedError

        if spin is None:
            if self.density.nspins == 1:
                nt_G = nt_sG[0]
            else:
                nt_G = nt_sG.sum(axis=0)
        else:
            if self.density.nspins == 1:
                nt_G = 0.5 * nt_sG[0]
            else:
                nt_G = nt_sG[spin]

        nt_G = gd.collect(nt_G, broadcast)

        if nt_G is None:
            return None
        
        if pad:
            nt_G = gd.zero_pad(nt_G)

        return nt_G / Bohr**3

    get_pseudo_valence_density = get_pseudo_density  # Don't use this one!
    
    def get_effective_potential(self, spin=0, pad=True, broadcast=False):
        """Return pseudo effective-potential."""
        # XXX should we do a gd.collect here?
        vt_G = self.hamiltonian.gd.collect(self.hamiltonian.vt_sG[spin],
                                           broadcast=broadcast)
        if vt_G is None:
            return None
        
        if pad:
            vt_G = self.hamiltonian.gd.zero_pad(vt_G)
        return vt_G * Hartree
    
    def get_pseudo_density_corrections(self):
        """Integrated density corrections.

        Returns the integrated value of the difference between the pseudo-
        and the all-electron densities at each atom.  These are the numbers
        you should add to the result of doing e.g. Bader analysis on the
        pseudo density."""
        if self.wfs.nspins == 1:
            return np.array([self.density.get_correction(a, 0)
                             for a in range(len(self.atoms))])
        else:
            return np.array([[self.density.get_correction(a, spin)
                              for a in range(len(self.atoms))]
                             for spin in range(2)])

    def get_all_electron_density(self, spin=None, gridrefinement=2,
                                 pad=True, broadcast=True, collect=True):
        """Return reconstructed all-electron density array."""
        n_sG, gd = self.density.get_all_electron_density(
            self.atoms, gridrefinement=gridrefinement)

        if spin is None:
            if self.density.nspins == 1:
                n_G = n_sG[0]
            else:
                n_G = n_sG.sum(axis=0)
        else:
            if self.density.nspins == 1:
                n_G = 0.5 * n_sG[0]
            else:
                n_G = n_sG[spin]

        if collect:
            n_G = gd.collect(n_G, broadcast)

        if n_G is None:
            return None
        
        if pad:
            n_G = gd.zero_pad(n_G)

        return n_G / Bohr**3

    def get_fermi_level(self):
        """Return the Fermi-level(s)."""
        eFermi = self.occupations.get_fermi_level()
        if eFermi is not None:
            eFermi *= Hartree
        return eFermi

    def get_fermi_levels(self):
        """Return the Fermi-levels in case of fixed-magmom."""
        eFermi_np_array = self.occupations.get_fermi_levels()
        if eFermi_np_array is not None:
            eFermi_np_array *= Hartree
        return eFermi_np_array

    def get_fermi_levels_mean(self):
        """Return the mean of th Fermi-levels in case of fixed-magmom."""
        eFermi_mean = self.occupations.get_fermi_levels_mean()
        if eFermi_mean is not None:
            eFermi_mean *= Hartree
        return eFermi_mean

    def get_fermi_splitting(self):
        """Return the Fermi-level-splitting in case of fixed-magmom."""
        eFermi_splitting = self.occupations.get_fermi_splitting()
        if eFermi_splitting is not None:
            eFermi_splitting *= Hartree
        return eFermi_splitting

    def get_wigner_seitz_densities(self, spin):
        """Get the weight of the spin-density in Wigner-Seitz cells
        around each atom.

        The density assigned to each atom is relative to the neutral atom,
        i.e. the density sums to zero.
        """
        from gpaw.analyse.wignerseitz import wignerseitz
        atom_index = wignerseitz(self.wfs.gd, self.atoms)

        nt_G = self.density.nt_sG[spin]
        weight_a = np.empty(len(self.atoms))
        for a in range(len(self.atoms)):
            # XXX Optimize! No need to integrate in zero-region
            smooth = self.wfs.gd.integrate(np.where(atom_index == a,
                                                    nt_G, 0.0))
            correction = self.density.get_correction(a, spin)
            weight_a[a] = smooth + correction
            
        return weight_a

    def get_dos(self, spin=0, npts=201, width=None):
        """The total DOS.

        Fold eigenvalues with Gaussians, and put on an energy grid.

        returns an (energies, dos) tuple, where energies are relative to the
        vacuum level for non-periodic systems, and the average potential for
        periodic systems.
        """
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        w_k = self.wfs.weight_k
        Nb = self.wfs.bd.nbands
        energies = np.empty(len(w_k) * Nb)
        weights = np.empty(len(w_k) * Nb)
        x = 0
        for k, w in enumerate(w_k):
            energies[x:x + Nb] = self.get_eigenvalues(k, spin)
            weights[x:x + Nb] = w
            x += Nb
            
        from gpaw.utilities.dos import fold
        return fold(energies, weights, npts, width)

    def get_wigner_seitz_ldos(self, a, spin=0, npts=201, width=None):
        """The Local Density of States, using a Wigner-Seitz basis function.

        Project wave functions onto a Wigner-Seitz box at atom ``a``, and
        use this as weight when summing the eigenvalues."""
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        from gpaw.utilities.dos import raw_wignerseitz_LDOS, fold
        energies, weights = raw_wignerseitz_LDOS(self, a, spin)
        return fold(energies * Hartree, weights, npts, width)
    
    def get_orbital_ldos(self, a,
                         spin=0, angular='spdf', npts=201, width=None):
        """The Local Density of States, using atomic orbital basis functions.

        Project wave functions onto an atom orbital at atom ``a``, and
        use this as weight when summing the eigenvalues.

        The atomic orbital has angular momentum ``angular``, which can be
        's', 'p', 'd', 'f', or any combination (e.g. 'sdf').

        An integer value for ``angular`` can also be used to specify a specific
        projector function to project onto.
        """
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        from gpaw.utilities.dos import raw_orbital_LDOS, fold
        energies, weights = raw_orbital_LDOS(self, a, spin, angular)
        return fold(energies * Hartree, weights, npts, width)

    def get_all_electron_ldos(self, mol, spin=0, npts=201, width=None,
                              wf_k=None, P_aui=None, lc=None, raw=False):
        """The Projected Density of States, using all-electron wavefunctions.

        Projects onto a pseudo_wavefunctions (wf_k) corresponding to some band
        n and uses P_aui ([paw.nuclei[a].P_uni[:,n,:] for a in atoms]) to
        obtain the all-electron overlaps.
        Instead of projecting onto a wavefunctions a molecular orbital can
        be specified by a linear combination of weights (lc)
        """
        from gpaw.utilities.dos import all_electron_LDOS, fold

        if raw:
            return all_electron_LDOS(self, mol, spin, lc=lc,
                                     wf_k=wf_k, P_aui=P_aui)
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        energies, weights = all_electron_LDOS(self, mol, spin,
                                              lc=lc, wf_k=wf_k, P_aui=P_aui)
        return fold(energies * Hartree, weights, npts, width)

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0, broadcast=True,
                                 pad=True):
        """Return pseudo-wave-function array.

        Units: 1/Angstrom^(3/2)
        """
        if pad:
            psit_G = self.get_pseudo_wave_function(band, kpt, spin, broadcast,
                                                   pad=False)
            if psit_G is None:
                return
            else:
                return self.wfs.gd.zero_pad(psit_G)
        psit_G = self.wfs.get_wave_function_array(band, kpt, spin)
        if broadcast:
            if not self.wfs.world.rank == 0:
                psit_G = self.wfs.gd.empty(dtype=self.wfs.dtype,
                                           global_array=True)
            self.wfs.world.broadcast(psit_G, 0)
            return psit_G / Bohr**1.5
        elif self.wfs.world.rank == 0:
            return psit_G / Bohr**1.5

    def get_eigenvalues(self, kpt=0, spin=0, broadcast=True):
        """Return eigenvalue array."""
        eps_n = self.wfs.collect_eigenvalues(kpt, spin)
        if broadcast:
            if self.wfs.world.rank != 0:
                assert eps_n is None
                eps_n = np.empty(self.wfs.bd.nbands)
            self.wfs.world.broadcast(eps_n, 0)
        if eps_n is not None:
            return eps_n * Hartree

    def get_occupation_numbers(self, kpt=0, spin=0, broadcast=True):
        """Return occupation array."""
        f_n = self.wfs.collect_occupations(kpt, spin)
        if broadcast:
            if self.wfs.world.rank != 0:
                assert f_n is None
                f_n = np.empty(self.wfs.bd.nbands)
            self.wfs.world.broadcast(f_n, 0)
        return f_n
    
    def get_xc_difference(self, xc):
        if isinstance(xc, str):
            xc = XC(xc)
        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)
        xc.set_positions(self.atoms.get_scaled_positions() % 1.0)
        if xc.orbital_dependent:
            self.converge_wave_functions()
        return self.hamiltonian.get_xc_difference(xc, self.density) * Hartree

    def initial_wannier(self, initialwannier, kpointgrid, fixedstates,
                        edf, spin, nbands):
        """Initial guess for the shape of wannier functions.

        Use initial guess for wannier orbitals to determine rotation
        matrices U and C.
        """
        #if not self.wfs.gamma:
        #    raise NotImplementedError
        from ase.dft.wannier import rotation_from_projection
        proj_knw = self.get_projections(initialwannier, spin)
        U_kww = []
        C_kul = []
        for fixed, proj_nw in zip(fixedstates, proj_knw):
            U_ww, C_ul = rotation_from_projection(proj_nw[:nbands],
                                                  fixed,
                                                  ortho=True)
            U_kww.append(U_ww)
            C_kul.append(C_ul)

        U_kww = np.asarray(U_kww)
        return C_kul, U_kww

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        k_kc = self.wfs.kd.bzk_kc
        G_c = k_kc[nextkpoint] - k_kc[kpoint] - G_I

        return self.get_wannier_integrals(spin, kpoint,
                                          nextkpoint, G_c, nbands)

    def get_wannier_integrals(self, s, k, k1, G_c, nbands=None):
        """Calculate integrals for maximally localized Wannier functions."""

        assert s <= self.wfs.nspins
        kpt_rank, u = divmod(k + len(self.wfs.kd.ibzk_kc) * s,
                             len(self.wfs.kpt_u))
        kpt_rank1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * s,
                               len(self.wfs.kpt_u))
        kpt_u = self.wfs.kpt_u

        # XXX not for the kpoint/spin parallel case
        assert self.wfs.kpt_comm.size == 1

        # If calc is a save file, read in tar references to memory
        self.wfs.initialize_wave_functions_from_restart_file()
        
        # Get pseudo part
        Z_nn = self.wfs.gd.wannier_matrix(kpt_u[u].psit_nG,
                                          kpt_u[u1].psit_nG, G_c, nbands)

        # Add corrections
        self.add_wannier_correction(Z_nn, G_c, u, u1, nbands)

        self.wfs.gd.comm.sum(Z_nn)
            
        return Z_nn

    def add_wannier_correction(self, Z_nn, G_c, u, u1, nbands=None):
        """
        Calculate the correction to the wannier integrals Z,
        given by (Eq. 27 ref1)::

                          -i G.r
            Z   = <psi | e      |psi >
             nm       n             m
                            
                           __                __
                   ~      \              a  \     a*   a    a
            Z    = Z    +  ) exp[-i G . R ]  )   P   dO    P
             nmx    nmx   /__            x  /__   ni   ii'  mi'

                           a                 ii'

        Note that this correction is an approximation that assumes the
        exponential varies slowly over the extent of the augmentation sphere.

        ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005)
        """

        if nbands is None:
            nbands = self.wfs.bd.nbands
            
        P_ani = self.wfs.kpt_u[u].P_ani
        P1_ani = self.wfs.kpt_u[u1].P_ani
        spos_ac = self.atoms.get_scaled_positions()
        for a, P_ni in P_ani.items():
            P_ni = P_ani[a][:nbands]
            P1_ni = P1_ani[a][:nbands]
            dO_ii = self.wfs.setups[a].dO_ii
            e = np.exp(-2.j * np.pi * np.dot(G_c, spos_ac[a]))
            Z_nn += e * np.dot(np.dot(P_ni.conj(), dO_ii), P1_ni.T)
            
    def get_projections(self, locfun, spin=0):
        """Project wave functions onto localized functions

        Determine the projections of the Kohn-Sham eigenstates
        onto specified localized functions of the format::

          locfun = [[spos_c, l, sigma], [...]]

        spos_c can be an atom index, or a scaled position vector. l is
        the angular momentum, and sigma is the (half-) width of the
        radial gaussian.

        Return format is::

          f_kni = <psi_kn | f_i>

        where psi_kn are the wave functions, and f_i are the specified
        localized functions.

        As a special case, locfun can be the string 'projectors', in which
        case the bound state projectors are used as localized functions.
        """

        wfs = self.wfs
        
        if locfun == 'projectors':
            f_kin = []
            for kpt in wfs.kpt_u:
                if kpt.s == spin:
                    f_in = []
                    for a, P_ni in kpt.P_ani.items():
                        i = 0
                        setup = wfs.setups[a]
                        for l, n in zip(setup.l_j, setup.n_j):
                            if n >= 0:
                                for j in range(i, i + 2 * l + 1):
                                    f_in.append(P_ni[:, j])
                            i += 2 * l + 1
                    f_kin.append(f_in)
            f_kni = np.array(f_kin).transpose(0, 2, 1)
            return f_kni.conj()

        from gpaw.lfc import LocalizedFunctionsCollection as LFC
        from gpaw.spline import Spline
        from gpaw.utilities import _fact

        nkpts = len(wfs.kd.ibzk_kc)
        nbf = np.sum([2 * l + 1 for pos, l, a in locfun])
        f_kni = np.zeros((nkpts, wfs.bd.nbands, nbf), wfs.dtype)

        spos_ac = self.atoms.get_scaled_positions() % 1.0
        spos_xc = []
        splines_x = []
        for spos_c, l, sigma in locfun:
            if isinstance(spos_c, int):
                spos_c = spos_ac[spos_c]
            spos_xc.append(spos_c)
            alpha = .5 * Bohr**2 / sigma**2
            r = np.linspace(0, 10. * sigma, 500)
            f_g = (_fact[l] * (4 * alpha)**(l + 3 / 2.) *
                   np.exp(-alpha * r**2) /
                   (np.sqrt(4 * np.pi) * _fact[2 * l + 1]))
            splines_x.append([Spline(l, rmax=r[-1], f_g=f_g)])
            
        lf = LFC(wfs.gd, splines_x, wfs.kd, dtype=wfs.dtype)
        lf.set_positions(spos_xc)

        assert wfs.gd.comm.size == 1
        k = 0
        f_ani = lf.dict(wfs.bd.nbands)
        for kpt in wfs.kpt_u:
            if kpt.s != spin:
                continue
            lf.integrate(kpt.psit_nG[:], f_ani, kpt.q)
            i1 = 0
            for x, f_ni in f_ani.items():
                i2 = i1 + f_ni.shape[1]
                f_kni[k, :, i1:i2] = f_ni
                i1 = i2
            k += 1

        return f_kni.conj()

    def get_number_of_grid_points(self):
        return self.wfs.gd.N_c

    def get_number_of_iterations(self):
        return self.niterations

    def get_ensemble_coefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.get_potential_energy()
        E0 = self.get_xc_difference('XC-9-1.0')
        coefs = (E + E0,
                 self.get_xc_difference('XC-0-1.0') - E0,
                 self.get_xc_difference('XC-1-1.0') - E0,
                 self.get_xc_difference('XC-2-1.0') - E0)
        self.text('BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs)
        return np.array(coefs)

    def get_electronic_temperature(self):
        # XXX do we need this - yes we do!
        return self.occupations.width * Hartree

    def get_number_of_electrons(self):
        return self.wfs.setups.nvalence - self.density.charge

    def get_electrostatic_corrections(self):
        """Calculate PAW correction to average electrostatic potential."""
        dEH_a = np.zeros(len(self.atoms))
        for a, D_sp in self.density.D_asp.items():
            setup = self.wfs.setups[a]
            dEH_a[a] = setup.dEH0 + np.dot(setup.dEH_p, D_sp.sum(0))
        self.wfs.gd.comm.sum(dEH_a)
        return dEH_a * Hartree * Bohr**3

    def read_wave_functions(self, mode='gpw'):
        """Read wave functions one by one from separate files"""

        from gpaw.io import read_wave_function
        for u, kpt in enumerate(self.wfs.kpt_u):
            #kpt = self.kpt_u[u]
            kpt.psit_nG = self.wfs.gd.empty(self.wfs.bd.nbands, self.wfs.dtype)
            # Read band by band to save memory
            s = kpt.s
            k = kpt.k
            for n, psit_G in enumerate(kpt.psit_nG):
                psit_G[:] = read_wave_function(self.wfs.gd, s, k, n, mode)

    def get_nonselfconsistent_energies(self, type='beefvdw'):
        from gpaw.xc.bee import BEEF_Ensemble
        if type not in ['beefvdw', 'mbeef']:
            raise NotImplementedError('Not implemented for type = %s' % type)
        assert self.scf.converged
        bee = BEEF_Ensemble(self)
        x = bee.create_xc_contributions('exch')
        c = bee.create_xc_contributions('corr')
        if type is 'beefvdw':
            return np.append(x,c)
        elif type is 'mbeef':
            return x.flatten()
