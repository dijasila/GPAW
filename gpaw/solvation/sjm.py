import traceback
from io import StringIO
import numbers
import numpy as np
import copy
import inspect

from ase.units import Bohr, Ha
from ase.calculators.calculator import Parameters, equal
from ase.dft.bandgap import bandgap

from gpaw.jellium import Jellium, JelliumSlab
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.fd_operators import Gradient
from gpaw.dipole_correction import DipoleCorrection

from gpaw.solvation.cavity import (Potential, Power12Potential,
                                   get_pbc_positions)
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from gpaw.solvation.poisson import WeightedFDPoissonSolver


def get_traceback_string():
    # FIXME. Temporary function for debugging.
    # (Will delete right before merge request.)
    # I guess traceback.format_stack would work easier!
    filelike = StringIO()
    traceback.print_stack(file=filelike)
    return filelike.getvalue()


class SJM(SolvationGPAW):
    """Solvated Jellium method.
    (Implemented as a subclass of the SolvationGPAW class.)

    The method allows the simulation of an electrochemical environment,
    where the potential can be varied by changing the charging (that is,
    number of electrons) of the system. For this purpose, it allows the usage
    of non-neutral periodic slab systems. Cell neutrality is achieved by adding
    a background charge in the solvent region above the slab

    Further details are given in http://dx.doi.org/10.1021/acs.jpcc.8b02465

    The method can be run in two modes:

        - Constant charge: The number of excess electrons in the simulation
          can be directly specified with the `ne` keyword, leaving
          target_potential=None.
        - Constant potential: The target potential (expressed as a work
          function) can be specified with the `target_potential` keyword.
          Optionally, the `ne` keyword can also be used to specify the initial
          guess of the number of electrons.

    By default, this method writes the grand-potential energy to the output;
    that is, the energy that has been adjusted with "- mu N" (in this case,
    mu is the workfunction and N is the excess electrons). This is the energy
    that is compatible with the forces in constant-potential mode and thus
    will behave well with optimizers, NEBs, etc. It is also frequently used in
    subsequent free-energy calculations.

    The SJM class takes a single argument, the sj dictionary. All other
    arguments are fed to the parent SolvationGPAW (and therefore GPAW)
    calculator.

    Parameters:

    sj: dict
        Dictionary of parameters for the solvated jellium method, whose
        possible keys are given below.

    Parameters in sj dictionary:

    ne: float  FIXME should this be 'excess_electrons' if we are going
    verbose on other names?????
        Number of electrons added in the atomic system and (with opposite
        sign) in the background charge region. If the `potential` keyword is
        also supplied, `ne` is taken as  an initial guess for the needed
        number of electrons.
    target_potential: float
        The potential that should be reached or kept in the course of the
        calculation. If set to "None" (default) a constant-charge
        calculation based on the value of `ne` is performed.
        Expressed as a workfunction, on the top side of the slab.
    tol: float
        Tolerance for the deviation of the target potential.
        If the potential is outside the defined range `ne` will be
        changed in order to get inside again. Default: 0.01 V.
    max_iters: int
        In constant-potential mode, the maximum number of iterations to try
        to equilibrate the potential (by varying ne). Default: 10.
    jelliumregion: dict FIXME shall we write "or None" (to get defaults)
        Parameters regarding the shape of the counter charge region
        Implemented keys:

        'upper_limit': float
            Upper boundary of the counter charge region in terms of
            coordinate in Angstrom (default: z).
            Default: atoms.cell[2][2] - 1.
        'lower_limit': float or 'cavity_like'
            If a float is given it corresponds to the lower
            boundary coordinate (default: z), where the counter charge
            starts. If 'cavity_like' is given the counter charge will
            take the form of the cavity up to the 'upper_limit'.
            Default: Highest z-coordinate + 3
        'thickness': float
            Thickness of the counter charge region in Angstrom.
            Can only be used if start is not 'cavity_like' and will
            be overwritten by 'upper_limit'. Default: None
        'fix_lower_limit': bool
            Do not allow the lower limit of the jellium region to adapt
            to a change in the atomic geometry during a relaxation.
            Omitted in a 'cavity_like' jelliumregion.
            Default: False

    verbose: bool or 'cube'
        True:
            Write final electrostatic potential, background charge and
            and cavity into ASCII files.
        'cube':
            In addition to 'True', also write the cavity on the
            3D-grid into a cube file.
    write_grandpotential_energy: bool
        Write the grand-potential energy into output files such as
        trajectory files. Default: True
    always_adjust: bool
        Adjust ne again even when potential is within tolerance.
        This is useful to set to True along with a loose potential
        tolerance (tol) to allow the potential and structure to be
        simultaneously optimized in a geometry optimization, for example.
        Default: False.
    """
    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms',
                              'excess_electrons', 'electrode_potential']

    default_parameters = copy.deepcopy(SolvationGPAW.default_parameters)
    default_parameters.update(
        {'sj': {'excess_electrons': 0.,
                'jelliumregion': None,  # FIXME: put in defaults?
                'target_potential': None,
                'write_grandpotential_energy': True,
                'tol': 0.01,
                'always_adjust': False,  # FIXME: rename to not use 'ne'.
                'verbose': False,
                'max_iters': 10.}})

    def __init__(self, restart=None, **gpaw_kwargs):
        # FIXME/ap: change **gpaw_kwargs to **kwargs?

        deprecated_keys = ['ne', 'potential', 'write_grandcanonical_energy',
                           'potential_equilibration_mode', 'dpot',
                           'max_pot_deviation', 'doublelayer', 'verbose']
        msg = ('{:s} is no longer a supported keyword argument for the SJM '
               'class. All SJM arguments should be sent in via the "sj" '
               'dict.')
        for key in deprecated_keys:
            if key in gpaw_kwargs:
                raise TypeError(msg.format(key))

        SolvationGPAW.__init__(self, restart, **gpaw_kwargs)
        # Note the previous line calls self.set().
        p = self.parameters['sj']  # Local parameters.

        # FIXME/ap: do we really want to hold local variables like the
        # slope inside of the parameters dictionary? I think everything
        # else in here is user-defined. Not sure if it creates any
        # problems.
        p.slope = None

        # FIXME/ap: Need to add back restart stuff, unless it's automagic
        # now that we're using GPAW's parameter scheme.

        self.sog('Solvated jellium method (SJM):')
        if not p.target_potential:
            # FIXME/ap: This log message seems to show up later, too, in
            # calculate. We can delete one or the other. Let's delete the
            # one that makes it consistent with how the code is structured
            # for constant-potential mode.
            self.sog(' Constant-charge mode. Excess electrons: {:.5f}'
                     .format(p.excess_electrons))
        else:
            self.sog(' Constant-potential mode.')
            self.sog(' Target potential: {:.5f} +/- {:.5f}'
                     .format(p.target_potential, p.tol))
            if not 2. < p.target_potential < 7.:
                self.sog('Warning! Your target potential might be '
                         'unphysical. Keep in mind 0 V_SHE is ~4.4 '
                         'in the input')
            self.sog(' Initial guess of excess electrons: {:.5f}'
                     .format(p.excess_electrons))

    def set(self, **kwargs):
        """Change parameters for calculator.

        It differs from the standard `set` function in two ways:
        - SJM specific keywords can be set
        - It does not reinitialize and delete `self.wfs` if the
          background charge is changed.

        """
        self.sog('SET INVOKED with kwargs:')
        self.sog(str(kwargs))

        old_sj_parameters = copy.deepcopy(self.parameters['sj'])

        parent_changed = False  # if something changed outside the SJ wrapper
        sj_dict = kwargs['sj'] if 'sj' in kwargs else {}
        badkeys = [key for key in sj_dict if key not in
                   self.default_parameters['sj']]
        if badkeys:
            raise KeyError('Unexpected key(s) "{}" provided to sj dict. '
                           'Only keys allowed are "{}".' .format(
                               ', '.join(badkeys),
                               ', '.join(self.default_parameters['sj'])))

        sj_changes = [key for key, value in sj_dict.items() if not
                      equal(value, self.parameters['sj'].get(key))]
        # GK: We exclude both sj keywords and background_charge from the
        #     keyword that should triger a initialization. background_charge
        #     is handled internally.
        if any(key not in ['sj', 'background_charge'] for key in kwargs):
            parent_changed = True
            SolvationGPAW.set(self, **kwargs)

        if not isinstance(self.parameters['sj'], Parameters):
            self.parameters['sj'] = Parameters(self.parameters['sj'])
        p = self.parameters['sj']
        # ase's Calculator.set() loses the dict items not being set.
        # Add them back.
        for key, value in old_sj_parameters.items():
            if key not in p:
                p[key] = value

        # FIXME: make sure all key names are still right.
        if 'target_potential' in sj_changes:
            self.results = {}
            # GK: p does not seem to update if it is not done explicitly
            # here
            p.target_potential = sj_dict['target_potential']
            if p.target_potential is None:
                self.sog('Potential equilibration has been turned off')
            else:
                self.sog('Target electrode potential set to {:1.4f} V'
                         .format(p.target_potential))

        if 'jelliumregion' in sj_changes:
            self.results = {}
            self.sog(' Jellium size parameters:')
            if 'start' in p.jelliumregion:
                # FIXME/ap: Since 'start' is no longer a keyword, should we
                # change this?
                self.sog('  Lower boundary: %s' % p.jelliumregion['start'])
            if 'upper_limit' in p.jelliumregion:
                self.sog('  Upper boundary: {:.3f} Angstroms'
                         .format(p.jelliumregion['upper_limit']))
            if self.atoms:
                # FIXME/AP: Is this needed here? Isn't called everytime in
                # self.calculate, and since the results are {}'d out above
                # it will be called then?
                # GK: That's what I did before, but now it seems having
                #     an empty results dictionary is not triggering a
                #     calculation anymore. So I am triggering it by using
                #     define_jellium ;).
                self.set(background_charge=self.define_jellium(self.atoms))

        if 'excess_electrons' in sj_changes:
            self.results = {}

            p.excess_electrons = sj_dict['excess_electrons']
            self.sog('Number of excess electrons manually changed '
                     'to {:1.8f}'.format(p.excess_electrons))
            if self.atoms is not None:
                self.set(background_charge=self.define_jellium(self.atoms))

        if 'tol' in sj_changes:
            p.tol = sj_dict['tol']
            self.sog('Potential tolerance has been changed to %1.4f V'
                     % p.tol)
            try:
                true_potential = self.get_electrode_potential()
            except AttributeError:
                pass
            else:
                if abs(true_potential - p.target_potential) > p.tol:
                    self.results = {}
                    self.sog('Recalculating...\n')
                    if self.atoms is not None:
                        p.excess_electrons = ((p.target_potential -
                                              true_potential) / p.slope)
                        self.set(background_charge=self.define_jellium(
                            self.atoms))
                else:
                    self.sog('Potential already reached the criterion.\n')

        if 'background_charge' in kwargs:
            # background_charge is a GPAW parameter.
            if self.wfs is None:
                SolvationGPAW.set(self, **kwargs)
            else:
                if parent_changed:
                    self.density = None
                else:
                    if self.density.nct_G is None:
                        self.initialize_positions()

                    self.density.reset()

                    if self.density.background_charge:
                        self.density.background_charge = \
                            kwargs['background_charge']
                        self.density.background_charge.set_grid_descriptor(
                            self.density.finegd)

                    self.spos_ac = self.atoms.get_scaled_positions() % 1.0
                    self.density.mixer.reset()
                    self.wfs.initialize(self.density, self.hamiltonian,
                                        self.spos_ac)
                    self.wfs.eigensolver.reset()
                    self.scf.reset()

                self.wfs.nvalence = (self.setups.nvalence +
                                     p.excess_electrons)
                self.sog('Number of valence electrons is now {:.5f}'
                         .format(self.wfs.nvalence))
                # FIXME: background_charge not always called when
                # excess_electrons changes? Doesn't this screw up nvalence
                # in wfs?
                # (I.e., see below where key == 'excess_electrons'.)
                # GK: Have think about this one, but I'm fairly certain
                # it was needed in order to avoid what you mentioned
                # AP: Ok, I don't fully get what's going on in this part,
                # so please check that it's doing what you want with the
                # new layout.

    def sog(self, message=''):
        # FIXME: Delete after all is set up.
        message = 'SJ: ' + str(message)
        self.log(message)
        self.log.flush()

    def create_hamiltonian(self, realspace, mode, xc):
        """
        This differs from SolvationGPAW's create_hamiltonian
        method by the ability to use dipole corrections.
        """
        if not realspace:
            raise NotImplementedError(
                'SJM does not support calculations in reciprocal space yet'
                ' due to a lack of an implicit solvent module.')

        dens = self.density

        self.hamiltonian = SJM_RealSpaceHamiltonian(
            *self.stuff_for_hamiltonian,
            gd=dens.gd, finegd=dens.finegd,
            nspins=dens.nspins,
            collinear=dens.collinear,
            setups=dens.setups,
            timer=self.timer,
            xc=xc,
            world=self.world,
            redistributor=dens.redistributor,
            vext=self.parameters.external,
            psolver=self.parameters.poissonsolver,
            stencil=mode.interpolation)

        xc.set_grid_descriptor(self.hamiltonian.finegd)

    # GK: I had to add this again. How did it work without??
    # AP: self._set_atoms is in gpaw.calculator.GPAW.
    # Maybe there's a syntax change in ASE calculators?
    # I've XXX'd it out and it works on my install; if it works
    # for you we'll delete.
    # FIXME/ap: Delete when everything works.
    def XXXset_atoms(self, atoms):
        self.atoms = atoms
        self.spos_ac = atoms.get_scaled_positions() % 1.0

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['cell']):
        """
        Perform a calculation with SJM.

        This method includes the potential equilibration loop.
        It is essentially a wrapper around GPAW.calculate()

        """

        if atoms is not None:
            # Need to be set before ASE's Calculator.calculate gets to it.
            self.atoms = atoms.copy()

        # FIXME: Should results be zerod when set is called? That might be
        # easiest. Then I can check here if the results are not zero
        # and we can calculate things directly.
        # GK: In general yes. However, some keywords might not need it
        # e.g. dpot
        p = self.parameters['sj']

        if not p.target_potential:
            self.sog('Constant-charge calculation with {:.5f} excess '
                     'electrons'.format(p.excess_electrons))
            # Background charge is set here, not earlier, because atoms needed.
            self.set(background_charge=self.define_jellium(atoms))
            SolvationGPAW.calculate(self, atoms, ['energy'], system_changes)
            self.sog('Potential found to be {:.5f} V (with {:+.5f} '
                     'electrons)'.format(self.get_electrode_potential(),
                                         p.excess_electrons))
            # FIXME: I think we can do a return here and cut out the next
            # else statement and indents? Or maybe not, there's junk at
            # the end?

        else:
            iteration = 0
            p.previous_electrons = []
            p.previous_potentials = []
            p.previous_slopes = []
            equilibrated = False

            while not equilibrated:
                self.sog('Attempt {:d} to equilibrate potential to {:.3f} +/-'
                         ' {:.3f} V'
                         .format(iteration, p.target_potential, p.tol))
                self.sog('Current guess of excess electrons: {:+.5f}'
                         .format(p.excess_electrons))
                if 0:
                    # GK: I took the following out, because I am not able
                    #     to trigger the error described. In case we can
                    #     turn it on again.
                    if iteration == 0:
                        # If we don't do this, GPAW.calculate *may* try to call
                        # self.density.reset(), which will not exist.
                        # FIXME/ap: to GK, I'm not clear on what things (like
                        # density, wfs, hamiltonian, scf) should be deleted
                        # and/or reset when. In the behavior here, gpaw is
                        # going to delete all of those at the beginning of
                        # every potential equilibration loop, but this seems
                        # wasteful. Note this is done when
                        # set(background_charge) is called.
                        system_changes += ['background_charge']
                if iteration == 1:
                    self.timer.start('Potential equilibration loop')
                    # We don't want SolvationGPAW to see any more system
                    # changes, like positions, after attempt 0.
                    system_changes = []

                # XXX: should it really be calling this every time?
                # Meaning: if the number of electrons did not change, then
                # I would think this would stay the same. Another question
                # is why it did not crash the first time through, shouldn't
                # density also be None then? But even given all that, we
                # need a way for this to work, as otherwise what happens
                # when a user changes ne or target potential manually? Oh,
                # I just tried that, let's see.
                if iteration > 0 or 'positions' in system_changes:
                    self.set(background_charge=self.define_jellium(atoms))

                SolvationGPAW.calculate(self, atoms, ['energy'],
                                        system_changes)
                true_potential = self.get_electrode_potential()
                self.sog()
                self.sog('Potential found to be {:.5f} V (with {:+.5f} '
                         'excess electrons, attempt {:d})'
                         .format(true_potential, p.excess_electrons,
                                 iteration))
                p.previous_electrons += [p.excess_electrons]
                p.previous_potentials += [true_potential]

                # Update slope based only on last two points.
                if len(p.previous_electrons) > 1:
                    p.slope = self.calculate_slope()
                    p.previous_slopes += [p.slope]
                    self.sog('Slope regressed from last two attempts is '
                             '{:.4f} V/electron.'.format(p.slope))

                    area = np.product(np.diag(atoms.cell[:2, :2]))
                    cap = -1.6022 * 1e3 / (area * p.slope)

                    self.sog('and apparent capacitance of {:.4f} muF/cm^2'
                             .format(cap))

                if abs(true_potential - p.target_potential) < p.tol:
                    equilibrated = True
                    self.sog('Potential is within tolerance. Equilibrated.')
                    if iteration >= 1:
                        self.timer.stop('Potential equilibration loop')
                    if p.always_adjust is False:
                        break

                # Change excess electrons based on slope.
                usr_warned = False
                if p.slope is None:
                    # FIXME/ap: Here's where we should use the electrode
                    # area to make this more robust, right?
                    self.sog('No slope information; changing electrons by 0.1'
                             ' to learn slope.')
                    p.excess_electrons += (0.1 * np.sign(true_potential -
                                           p.target_potential))
                else:
                    if true_potential < 2 and len(p.previous_potentials) > 1:
                        if np.diff(p.previous_potentials[-2:])[0] < -1:
                            true_potential = self.half_last_step()

                        elif ((p.target_potential - true_potential) /
                              p.slope) > 0:

                            self.sog(
                                '-' * 13 + '\n'
                                'Warning! Based on the slope the next step '
                                'will lead to very unphysical behavior\n '
                                '(workfunction ~ 0) and will likely '
                                'not converge.\n Recheck the initial guess '
                                'of the charges.\n' + '-' * 13)
                        usr_warned = True

                    p.excess_electrons += ((p.target_potential -
                                            true_potential) / p.slope)
                    self.sog('Number of electrons changed to {:.4f} based '
                             'on slope of {:.4f} V/electron.'
                             .format(p.excess_electrons, p.slope))

                if not 2 < true_potential < 7 and not usr_warned:
                    self.sog('-' * 13)
                    self.sog("Warning! The resulting potential is in a "
                             "region where a linear slope between\n potential "
                             "and charge is not guaranteed. You might want "
                             "to rethink\n the inital charge guess. If the "
                             "potential is too high increase "
                             "'excess_electrons' or vice versa.")
                    self.sog('-' * 13)

                iteration += 1
                if iteration == p.max_iters:
                    msg = ('Potential could not be reached after {:d} '
                           'iterations. This may indicate your workfunction '
                           'is noisier than tol. You may try setting the '
                           'convergence["workfunction"] keyword in GPAW. '
                           'Aborting!'.format(iteration))
                    self.sog(msg)
                    raise Exception(msg)

        if properties != ['energy']:
            SolvationGPAW.calculate(self, atoms, properties, [])

        # Note that grand-potential energies were assembled in summary,
        # which in turn was called by GPAW.calculate.

        if p.write_grandpotential_energy:
            self.results['energy'] = self.omega_extrapolated * Ha
            self.results['free_energy'] = self.omega_free * Ha
            self.sog('Grand-potential energy was written into results.\n')
        else:
            self.sog('Canonical energy was written into results.\n')

        self.results['excess_electrons'] = p.excess_electrons
        self.results['electrode_potential'] = self.get_electrode_potential()

        # FIXME: I believe that in the current version if you call
        #  atoms.get_potential_energy()
        #  atoms.get_forces()
        # Then the second line triggers a new SCF cycle even though
        # it should just pull the forces from the first.
        # I'm pretty sure this is because the line is invoked again:
        #        self.set(background_charge=self.define_jellium(atoms))
        # GK: Will test
        if p.verbose:
            self.write_cavity_and_bckcharge()

    def calculate_slope(self, coeff=1.):
        p = self.parameters['sj']
        return coeff * (np.diff(p.previous_potentials[-2:]) /
                        np.diff(p.previous_electrons[-2:]))[0]

    def half_last_step(self):
        p = self.parameters['sj']
        self.sog('-' * 13 + '\n'
                 'Warning! Your last step lead to a '
                 'dangerously low potential.\n '
                 'This is most likely due to overshooting the target.\n '
                 'The previous step size will be halved '
                 'and retried in order to avoid\n '
                 'unphysical behavior. In case the potential '
                 'will not equilibrate, \n rethink your '
                 'initial charge guess\n' + '-' * 13)

        del p.previous_potentials[-1]
        del p.previous_electrons[-1]

        self.wfs.nvalence = self.setups.nvalence + \
            p.previous_electrons[-2]
        p.slope = self.calculate_slope(coeff=2.)
        p.excess_electrons = p.previous_electrons[-1]
        return p.previous_potentials[-1]

    def write_cavity_and_bckcharge(self):
        p = self.parameters['sj']
        self.write_parallel_func_in_z(self.hamiltonian.cavity.g_g,
                                      name='cavity_')
        if p.verbose == 'cube':
            self.write_parallel_func_on_grid(self.hamiltonian.cavity.g_g,
                                             atoms=self.atoms,
                                             name='cavity')

        self.write_parallel_func_in_z(self.density.background_charge.mask_g,
                                      name='background_charge_')

    def summary(self):
        p = self.parameters['sj']
        efermi = self.wfs.fermi_level
        self.hamiltonian.summary(self.wfs, self.log)
        # Add grand-canonical terms.
        self.sog()
        mu_N = self.get_electrode_potential() * p.excess_electrons / Ha
        self.omega_free = self.hamiltonian.e_total_free + mu_N
        self.omega_extrapolated = self.hamiltonian.e_total_extrapolated + mu_N
        self.sog('Legendre-transformed energies (Omega = E - N mu)')
        self.sog('  (grand potential energies)')
        self.sog('  N (excess electrons): {:+11.6f}'
                 .format(p.excess_electrons))
        self.sog('  mu (workfunction, eV): {:+11.6f}'
                 .format(self.get_electrode_potential()))
        self.sog('-' * 26)
        self.sog('Free energy:   {:+11.6f}'.format(Ha * self.omega_free))
        self.sog('Extrapolated:  {:+11.6f}'
                 .format(Ha * self.omega_extrapolated))
        self.sog()

        # Back to standard output.
        self.density.summary(self.atoms, self.results.get('magmom', 0.),
                             self.log)
        self.wfs.summary(self.log)
        if len(self.wfs.fermi_levels) == 1:
            try:
                bandgap(self, output=self.log.fd, efermi=efermi * Ha)
            except ValueError:
                pass
        self.log.fd.flush()

        if p.verbose:
            self.write_parallel_func_in_z(self.hamiltonian.vHt_g * Ha -
                                          self.get_fermi_level(),
                                          'elstat_potential_')

    def define_jellium(self, atoms):
        """Method to define the explicit and counter charge."""

        p = self.parameters['sj']

        if np.isclose(p.excess_electrons, 0., atol=1e-5):
            return None

        if p.jelliumregion is None:
            p.jelliumregion = {}

        # FIXME/ap: From "The Zen of Python":
        #   There should be one-- and preferably only one --obvious way to
        #   do it.
        # Right now one can equivalently specify the (lower_limit +
        # thickness) or the (lower_limit + upper_limit) or the (upper_limit
        # + thickness), I think. Is there a reason to have so many ways to
        # define two boundaries? Can we drop a keyword? That would make our
        # lives easier below.

        if 'upper_limit' in p.jelliumregion:
            pass
        elif 'thickness' in p.jelliumregion:
            if p.jelliumregion['lower_limit'] == 'cavity_like':
                raise RuntimeError("With a cavity-like counter charge only "
                                   "the keyword 'upper_limit' (not "
                                   "'thickness') can be used.")
            else:
                p.jelliumregion['upper_limit'] = (p.jelliumregion['start'] +
                                                  p.jelliumregion['thickness'])
        else:
            p.jelliumregion['upper_limit'] = atoms.cell[2][2] - 1.0

        if p.jelliumregion['upper_limit'] > atoms.cell[2][2]:
            raise IOError('The upper limit of the jellium region lies '
                          'outside of unit cell. If you did not set it '
                          'manually, increase your unit cell size or '
                          'translate the atomic system down along the '
                          'z-axis.')

        if 'lower_limit' in p.jelliumregion:
            if p.jelliumregion['lower_limit'] == 'cavity_like':
                # XXX This part can definitely be improved
                if self.hamiltonian is None:
                    filename = self.log.fd
                    # self.log.fd = None  # FIXME This was causing crashes.
                    self.initialize(atoms)
                    self.set_positions(atoms)
                    # self.log.fd = filename
                    g_g = self.hamiltonian.cavity.g_g.copy()
                    self.wfs = None
                    self.density = None
                    self.hamiltonian = None
                    self.initialized = False

                else:
                    filename = self.log.fd
                    self.log.fd = None
                    self.set_positions(atoms)
                    self.log.fd = filename
                    g_g = self.hamiltonian.cavity.g_g

                return CavityShapedJellium(p.excess_electrons, g_g=g_g,
                                           z2=p.jelliumregion['upper_limit'])

            elif isinstance(p.jelliumregion['lower_limit'], numbers.Real):
                if p.jelliumregion['lower_limit'] > \
                        p.jelliumregion['upper_limit']:

                    raise RuntimeError('The lower limit of the jellium region '
                                       'lies above the upper limit. Recheck '
                                       'your input.')

                p.jelliumregion['start'] = p.jelliumregion['lower_limit']
            else:
                raise RuntimeError("The starting z value of the counter charge"
                                   "has to be either a number (coordinate),"
                                   "cavity_like' or not given (default: "
                                   " max(position)+3)")
        else:
            if 'fix_lower_limit' in p.jelliumregion.keys():
                if p.jelliumregion['fix_lower_limit']:
                    p.jelliumregion['lower_limit'] = \
                        max(atoms.positions[:, 2]) + 3.

            p.jelliumregion['start'] = max(atoms.positions[:, 2]) + 3.

        # FIXME/ap: We should get rid of the 'start' keyword I think, making
        # lower_limit.
        self.sog('Lower jellium boundary: %s' % p.jelliumregion['start'])
        self.sog('Upper jellium boundary: %s' % p.jelliumregion['upper_limit'])

        if p.jelliumregion['start'] > p.jelliumregion['upper_limit']:
            # FIXME/ap: This was checked once earlier, too? Can we cut one?
            raise IOError('The lower limit of the jellium region '
                          'lies above the upper limit. Increase your unit '
                          'cell size or translate the atomic system down '
                          'along the z-axis.')

        return JelliumSlab(p.excess_electrons, z1=p.jelliumregion['start'],
                           z2=p.jelliumregion['upper_limit'])

    def get_electrode_potential(self):
        """Returns the potential of the simulated electrode, in V, relative
        to the vacuum. This comes directly from the work function."""
        return self.hamiltonian.get_workfunctions(self.wfs.fermi_level)[1] * Ha

    """Various tools for writing global functions"""

    def write_parallel_func_in_z(self, g, name='g_z.out'):
        # FIXME: This needs some documentation!
        gd = self.density.finegd
        from gpaw.mpi import world
        G = gd.collect(g, broadcast=False)
        if world.rank == 0:
            G_z = G.mean(0).mean(0)
            name += '.'.join(self.log.fd.name.split('.')[:-1]) + '.out'
            out = open(name, 'w')
            # out = open(name + self.log.fd.name.split('.')[0] + '.out', 'w')
            for i, val in enumerate(G_z):
                out.writelines('%f  %1.8f\n' % ((i + 1) * gd.h_cv[2][2] * Bohr,
                               val))
            out.close()

    def write_parallel_func_on_grid(self, g, atoms=None, name='func.cube',
                                    outstyle='cube'):
        from ase.io import write
        gd = self.density.finegd
        G = gd.collect(g, broadcast=False)
        if outstyle == 'cube':
            write(name + '.cube', atoms, data=G)
        elif outstyle == 'pckl':
            # FIXME does this have parallel issues?
            # GK: Possibly have to check
            import pickle
            out = open(name, 'wb')
            pickle.dump(G, out)
            out.close()

    def _write(self, writer, mode):
        self.sog('in method ' + inspect.stack()[0][3])
        SolvationGPAW._write(self, writer, mode)
        writer.child('SJM').write(**self.parameters['sj'])

    def read(self, filename):
        self.sog('in method ' + inspect.stack()[0][3])
        reader = SolvationGPAW.read(self, filename)

        if 'SJM' in reader:
            self.parameters['sj'] = Parameters()
            for sj_key in reader.SJM.keys():
                self.parameters['sj'][sj_key] = reader.SJM.asdict()[sj_key]


class SJMPower12Potential(Power12Potential):
    # FIXME. We should state how this differs from its parent.
    # Also could it be integrated now that we are comfortable with SJM?
    # GK: This class is explicitly called in the SolvationGPAW keywords.
    # Thus, I guess it might be a little more of an effort to include it.

    """Inverse power law potential.

    An 1 / r ** 12 repulsive potential
    taking the value u0 at the atomic radius.

    See also
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """
    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, atomic_radii, u0, pbc_cutoff=1e-6, tiny=1e-10,
                 H2O_layer=False, unsolv_backside=True):
        """Constructor for the SJMPower12Potential class.
        In SJM one also has the option of removing the solvent from the
        electrode backside and adding ghost plane/atoms to remove the solvent
        from the electrode-water interface.

        Parameters
        ----------
        atomic_radii : float
            Callable mapping an ase.Atoms object to an iterable of atomic radii
            in Angstroms.
        u0 : float
            Strength of the potential at the atomic radius in eV.
        pbc_cutoff : float
            Cutoff in eV for including neighbor cells in a calculation with
            periodic boundary conditions.
        H2O_layer: bool,int or 'plane' (default: False)
            True: Exclude the implicit solvent from the interface region
                between electrode and water. Ghost atoms will be added below
                the water layer.
            int: Explicitly account for the given number of water molecules
                above electrode. This is handy if H2O is directly adsorbed
                and a water layer is present in the unit cell at the same time.
            'plane': Use a plane instead of ghost atoms for freeing the
                surface.
        unsolv_backside: bool
            Exclude implicit solvent from the region behind the electrode

        """
        Potential.__init__(self)
        self.atomic_radii = atomic_radii
        self.u0 = float(u0)
        self.pbc_cutoff = float(pbc_cutoff)
        self.tiny = float(tiny)
        self.r12_a = None
        self.r_vg = None
        self.pos_aav = None
        self.del_u_del_r_vg = None
        self.atomic_radii_output = None
        self.H2O_layer = H2O_layer
        self.unsolv_backside = unsolv_backside

    def write(self, writer):
        writer.write(
            name='SJMPower12Potential',
            u0=self.u0,
            atomic_radii=self.atomic_radii_output,
            H2O_layer=self.H2O_layer,
            unsolv_backside=self.unsolv_backside)

    def update(self, atoms, density):
        if atoms is None:
            return False
        self.r12_a = (self.atomic_radii_output / Bohr) ** 12
        r_cutoff = (self.r12_a.max() * self.u0 / self.pbc_cutoff) ** (1. / 12.)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.u_g.fill(.0)
        self.grad_u_vg.fill(.0)
        na = np.newaxis

        if self.unsolv_backside:
            # Removing solvent from electrode backside
            for z in range(self.u_g.shape[2]):
                if (self.r_vg[2, 0, 0, z] - atoms.positions[:, 2].min() /
                        Bohr < 0):
                    self.u_g[:, :, z] = np.inf
                    self.grad_u_vg[:, :, :, z] = 0

        if self.H2O_layer:
            # Add ghost coordinates and indices to pos_aav dictionary if
            # a water layer is present

            all_oxygen_ind = [atom.index for atom in atoms
                              if atom.symbol == 'O']

            # Disregard oxygens that don't belong to the water layer
            allwater_oxygen_ind = []
            for ox in all_oxygen_ind:
                nH = 0

                for i, atm in enumerate(atoms):
                    for period_atm in self.pos_aav[i]:
                        dist = period_atm * Bohr - atoms[ox].position
                        if np.linalg.norm(dist) < 1.3 and atm.symbol == 'H':
                            nH += 1

                if nH >= 2:
                    allwater_oxygen_ind.append(ox)

            # If the number of waters in the water layer is given as an input
            # (H2O_layer=i) then only the uppermost i water molecules are
            # regarded for unsolvating the interface (this is relevant if
            # water is adsorbed on the surface)
            if not isinstance(self.H2O_layer, (bool, str)):
                if self.H2O_layer % 1 < self.tiny:
                    self.H2O_layer = int(self.H2O_layer)
                else:
                    raise AttributeError('Only an integer number of water'
                                         'molecules is possible in the water'
                                         'layer')

                allwaters = atoms[allwater_oxygen_ind]
                indizes_water_ox_ind = np.argsort(allwaters.positions[:, 2],
                                                  axis=0)

                water_oxygen_ind = []
                for i in range(self.H2O_layer):
                    water_oxygen_ind.append(
                        allwater_oxygen_ind[indizes_water_ox_ind[-1 - i]])

            else:
                water_oxygen_ind = allwater_oxygen_ind

            oxygen = self.pos_aav[water_oxygen_ind[0]] * Bohr
            if len(water_oxygen_ind) > 1:
                for windex in water_oxygen_ind[1:]:
                    oxygen = np.concatenate(
                        (oxygen, self.pos_aav[windex] * Bohr))

            O_layer = []
            if isinstance(self.H2O_layer, str):
                # Add a virtual plane
                if len(self.H2O_layer.split('-')) > 1:
                    plane_z = float(self.H2O_layer.split('-')[1]) - \
                        1.0 * self.atomic_radii_output[water_oxygen_ind[0]]
                else:
                    plane_rel_oxygen = -1.5 * self.atomic_radii_output[
                        water_oxygen_ind[0]]
                    plane_z = oxygen[:, 2].min() + plane_rel_oxygen

                r_diff_zg = self.r_vg[2, :, :, :] - plane_z / Bohr
                r_diff_zg[r_diff_zg < self.tiny] = self.tiny
                r_diff_zg2 = r_diff_zg ** 2
                u_g = self.r12_a[water_oxygen_ind[0]] / r_diff_zg2 ** 6
                self.u_g += u_g.copy()
                u_g /= r_diff_zg2
                r_diff_zg *= u_g.copy()
                self.grad_u_vg[2, :, :, :] += r_diff_zg

            else:
                # Ghost atoms are added below the explicit water layer
                cell = atoms.cell.copy() / Bohr
                cell[2][2] = 1.
                natoms_in_plane = [round(np.linalg.norm(cell[0]) * 1.5),
                                   round(np.linalg.norm(cell[1]) * 1.5)]

                plane_z = (oxygen[:, 2].min() - 1.75 *
                           self.atomic_radii_output[water_oxygen_ind[0]])
                nghatoms_z = int(round(oxygen[:, 2].min() -
                                 atoms.positions[:, 2].min()))

                for i in range(int(natoms_in_plane[0])):
                    for j in range(int(natoms_in_plane[1])):
                        for k in np.linspace(atoms.positions[:, 2].min(),
                                             plane_z, num=nghatoms_z):

                            O_layer.append(np.dot(np.array(
                                [(1.5 * i - natoms_in_plane[0] / 4) /
                                 natoms_in_plane[0],
                                 (1.5 * j - natoms_in_plane[1] / 4) /
                                 natoms_in_plane[1],
                                 k / Bohr]), cell))

                # Add additional ghost O-atoms below the actual water O atoms
                # of water which frees the interface in case of corrugated
                # water layers
                for ox in oxygen / Bohr:
                    O_layer.append([ox[0], ox[1], ox[2] - 1.0 *
                                    self.atomic_radii_output[
                                        water_oxygen_ind[0]] / Bohr])

                r12_add = []
                for i in range(len(O_layer)):
                    self.pos_aav[len(atoms) + i] = [O_layer[i]]
                    r12_add.append(self.r12_a[water_oxygen_ind[0]])
                r12_add = np.array(r12_add)
                # r12_a must have same dimensions as pos_aav items
                self.r12_a = np.concatenate((self.r12_a, r12_add))

        for index, pos_av in self.pos_aav.items():
            pos_av = np.array(pos_av)
            r12 = self.r12_a[index]
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r_diff_vg = self.r_vg - origin_vg
                r_diff2_g = (r_diff_vg ** 2).sum(0)
                r_diff2_g[r_diff2_g < self.tiny] = self.tiny
                u_g = r12 / r_diff2_g ** 6
                self.u_g += u_g
                u_g /= r_diff2_g
                r_diff_vg *= u_g[na, ...]
                self.grad_u_vg += r_diff_vg

        self.u_g *= self.u0 / Ha
        self.grad_u_vg *= -12. * self.u0 / Ha
        self.grad_u_vg[self.grad_u_vg < -1e20] = -1e20
        self.grad_u_vg[self.grad_u_vg > 1e20] = 1e20

        return True


class SJM_RealSpaceHamiltonian(SolvationRealSpaceHamiltonian):
    # FIXME If the docs are right this just has a dipole added, so it could
    # be folded into SolvationGPAW's implementation, right?
    # GK: Yes, I think that is the only reason why it is a separate class
    """Realspace Hamiltonian with continuum solvent model in the context of
    SJM.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).

    In contrast to the standard implicit solvent model a dipole correction can
    also be applied.

    """

    def __init__(self, cavity, dielectric, interactions, gd, finegd, nspins,
                 setups, timer, xc, world, redistributor, vext=None,
                 psolver=None, stencil=3, collinear=None):
        """Constructor of SJM_RealSpaceHamiltonian class.


        Notes
        -----
        The only difference to SolvationRealSpaceHamiltonian is the
        possibility to perform a dipole correction

        """

        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        for ia in interactions:
            ia.set_grid_descriptor(finegd)

        if psolver is None:
            psolver = WeightedFDPoissonSolver()
            self.dipcorr = False
        elif isinstance(psolver, dict):
            psolver = SJMDipoleCorrection(WeightedFDPoissonSolver(),
                                          psolver['dipolelayer'])
            self.dipcorr = True

        if self.dipcorr:
            psolver.poissonsolver.set_dielectric(self.dielectric)
        else:
            psolver.set_dielectric(self.dielectric)

        self.gradient = None

        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, collinear, setups, timer, xc, world,
            vext=vext, psolver=psolver,
            stencil=stencil, redistributor=redistributor)

        for ia in interactions:
            setattr(self, 'e_' + ia.subscript, None)
        self.new_atoms = None
        self.vt_ia_g = None
        self.e_total_free = None
        self.e_total_extrapolated = None

    def initialize(self):
        if self.dipcorr:
            self.gradient = [Gradient(self.finegd, i, 1.0,
                             self.poisson.poissonsolver.nn)
                             for i in (0, 1, 2)]
        else:
            self.gradient = [Gradient(self.finegd, i, 1.0,
                             self.poisson.nn)
                             for i in (0, 1, 2)]

        self.vt_ia_g = self.finegd.zeros()
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        RealSpaceHamiltonian.initialize(self)


class CavityShapedJellium(Jellium):
    """The Solvated Jellium object, where the counter charge takes the form
       of the cavity.
    """
    def __init__(self, charge, g_g, z2):
        """Put the jellium background charge where the solvent is present and
           z < z2.

        Parameters:
        ----------
        charge: float
            The total jellium background charge.
        g_g: array
            The g function from the implicit solvent model, representing the
            percentage of the actual dielectric constant on the grid.
        z2: float
            Position of upper surface in Angstrom units.
        """

        Jellium.__init__(self, charge)
        self.g_g = g_g
        self.z2 = (z2 - 0.0001) / Bohr

    def todict(self):
        dct = Jellium.todict(self)
        dct.update(z2=self.z2 * Bohr + 0.0001)
        return dct

    def get_mask(self):
        r_gv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        mask = np.logical_not(r_gv[:, :, :, 2] > self.z2).astype(float)
        mask *= self.g_g
        return mask


class SJMDipoleCorrection(DipoleCorrection):
    # FIXME We should document why this needs to be different than the
    # standard version. It is not obvious to me.
    # GK: The standard dipole correction does not work for SJM. If used
    # the potential is not flat outside. The reason is that there's regions
    # with  varying dielectric properties in the cell. Thus, the dipole alone
    # is not enough for canceling the field at the outer regions, since field
    # is proportional to q/epsilon. Just scaling by espilon doesn't work
    # either, unfortunately, because of the mentioned inhomogenity in epsilon.
    # This is why it took me so long to get this working and even now it is
    # only working iteratively. At some point in the future this should be
    # revisited (again...) and solved in a more rigorous way. The keyword
    # here is "Polarization charges", I think.
    """Dipole-correcting wrapper around another PoissonSolver specific for SJM.

    Iterative dipole correction class as applied in SJM.

    Notes
    -----

    The modules can easily be incorporated in the trunk version of GPAW
    by just adding the `fd_solv_solve`  and adapting the `solve` modules
    in the `DipoleCorrection` class.

    This module is currently calculating the correcting dipole potential
    iteratively and we would be very grateful if anybody could
    provide an analytical solution.

    New Parameters
    ---------
    corrterm: float
    Correction factor for the added countering dipole. This is calculated
    iteratively.

    last_corrterm: float
    Corrterm in the last iteration for getting the change of slope with change
    corrterm

    last_slope: float
    Same as for `last_corrterm`

    """
    def __init__(self, poissonsolver, direction, width=1.0):
        """Construct dipole correction object."""

        DipoleCorrection.__init__(self, poissonsolver, direction, width=1.0)
        self.corrterm = 1
        self.elcorr = None
        self.last_corrterm = None

    def solve(self, pot, dens, **kwargs):
        if isinstance(dens, np.ndarray):
            # finite-diference Poisson solver:
            if hasattr(self.poissonsolver, 'dielectric'):
                return self.fd_solv_solve(pot, dens, **kwargs)
            else:
                return self.fdsolve(pot, dens, **kwargs)
        # Plane-wave solver:
        self.pwsolve(pot, dens)

    def fd_solv_solve(self, vHt_g, rhot_g, **kwargs):

        gd = self.poissonsolver.gd
        slope_lim = 1e-8
        slope = slope_lim * 10

        dipmom = gd.calculate_dipole_moment(rhot_g)[2]

        if self.elcorr is not None:
            vHt_g[:, :] -= self.elcorr

        iters2 = self.poissonsolver.solve(vHt_g, rhot_g, **kwargs)

        sawtooth_z = self.sjm_sawtooth()
        L = gd.cell_cv[2, 2]

        while abs(slope) > slope_lim:
            vHt_g2 = vHt_g.copy()
            self.correction = 2 * np.pi * dipmom * L / \
                gd.volume * self.corrterm
            elcorr = -2 * self.correction

            elcorr *= sawtooth_z
            elcorr2 = elcorr[gd.beg_c[2]:gd.end_c[2]]
            vHt_g2[:, :] += elcorr2

            VHt_g = gd.collect(vHt_g2, broadcast=True)
            VHt_z = VHt_g.mean(0).mean(0)
            slope = VHt_z[2] - VHt_z[10]

            if abs(slope) > slope_lim:
                if self.last_corrterm is not None:
                    ds = (slope - self.last_slope) / \
                        (self.corrterm - self.last_corrterm)
                    con = slope - (ds * self.corrterm)
                    self.last_corrterm = self.corrterm
                    self.corrterm = -con / ds
                else:
                    self.last_corrterm = self.corrterm
                    self.corrterm -= slope * 10.
                self.last_slope = slope
            else:
                vHt_g[:, :] += elcorr2
                self.elcorr = elcorr2

        return iters2

    def sjm_sawtooth(self):
        gd = self.poissonsolver.gd
        c = self.c
        L = gd.cell_cv[c, c]
        step = gd.h_cv[c, c] / L

        eps_g = gd.collect(self.poissonsolver.dielectric.eps_gradeps[0],
                           broadcast=True)
        eps_z = eps_g.mean(0).mean(0)

        saw = [-0.5]
        for i, eps in enumerate(eps_z):
            saw.append(saw[i] + step / eps)
        saw = np.array(saw)
        saw /= saw[-1] + step / eps_z[-1] - saw[0]
        saw -= (saw[0] + saw[-1] + step / eps_z[-1]) / 2.
        return saw
