import numpy as np
from gpaw.external import ConstantElectricField
from ase.units import alpha, Hartree, Bohr
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from scipy import interpolate
from gpaw.tddft.spectrum import read_td_file_kicks


class RRemission(object):
    r"""
    Radiation-reaction potential according to Schaefer et al.
    [https://doi.org/10.1103/PhysRevLett.128.156402] and
    Schaefer [https://doi.org/10.48550/arXiv.2204.01602].
    The potential accounts for the friction
    forces acting on the radiating system of oscillating charges
    emitting into a single dimension. A more elegant
    formulation would use the current instead of the dipole.
    Please contact christian.schaefer.physics@gmail.com if any problems
    should appear or you would like to consider more complex systems.
    Big thanks to Tuomas Rossi and Jakub Fojt for their help.

    Parameters
    ----------
    rr_quantization_plane: float
        value of :math:`rr_quantization_plane` in atomic units
    pol_cavity: array
        value of :math:`pol_cavity` dimensionless (directional)
    environmentcavity_in: array
        [0] lowest harmonic of cavity,
        [1] highest harmonic (cutoff),
        [2] loss of cavity, [3] dummy, [4] dummy,
        [5] increase frequency resolution
        comment: [3 and 4] are dummies that should be removed later
                 but are kept at the moment to keep my scripts running.
    environmentensemble_in: array
        [0] number of ensemble oscillators,
        [1] Ve/V ratio of occupied ensemble to cavity volume,
        [2] resonance frequency of Drude-Lorentz for ensemble,
        [3] lossyness of ensemble oscillators,
        [4] plasma-frequency of ensemble oscillators
        comment: If [2], [3] or [4] equal 0, the code will attempt to load
                 3 dipole files to compute the polarizability matrix.
                 In that case [1] is ignored.
    """

    def __init__(self, rr_quantization_plane_in, pol_cavity_in,
                 environmentcavity_in=None, environmentensemble_in=None):
        self.rr_quantization_plane = rr_quantization_plane_in / Bohr**2
        self.polarization_cavity = pol_cavity_in
        self.dipolexyz = None
        if environmentcavity_in is None:
            self.environment = 0
        else:
            self.environment = 1
            self.cavity_resonance = (np.arange(environmentcavity_in[0],
                                               environmentcavity_in[1] + 1e-8,
                                               environmentcavity_in[0])
                                     / Hartree)
            self.cavity_volume = (self.rr_quantization_plane
                                  * (np.pi / (alpha *
                                              self.cavity_resonance[0])))
            self.cavity_loss = (environmentcavity_in[2]
                                * self.cavity_resonance[0])
            self.deltat = None
            self.maxtimesteps = None
            self.frequ_resolution_ampl = environmentcavity_in[5]
            if environmentensemble_in is not None:
                self.ensemble_number = environmentensemble_in[0]
                self.ensemble_occupation_ratio = environmentensemble_in[1]
                self.ensemble_resonance = (environmentensemble_in[2]
                                           / Hartree)
                self.ensemble_loss = (environmentensemble_in[3]
                                      * self.ensemble_resonance)
                self.ensemble_plasmafrequency = (environmentensemble_in[4]
                                                 / Hartree)
            else:
                self.ensemble_number = environmentensemble_in

    def write(self, writer):
        writer.write(DelDipole=self.dipolexyz)

    def read(self, reader):
        self.dipolexyz = reader.DelDipole

    def initialize(self, paw):
        self.iterpredcop = 0
        if self.dipolexyz is None:
            self.dipolexyz = [0, 0, 0]
        self.density = paw.density
        self.wfs = paw.wfs
        self.hamiltonian = paw.hamiltonian
        self.dipolexyz_previous = self.density.calculate_dipole_moment()
        self.itert = 0
        if self.environment == 1:
            self.dyadic = None

    def vradiationreaction(self, kpt, time):
        if self.environment == 1 and self.dyadic is None:
            self.dyadic = self.dyadicGt(self.deltat, self.maxtimesteps)
            self.dipole_projected = np.zeros(self.maxtimesteps)
        if self.iterpredcop == 0:
            self.iterpredcop += 1
            self.dipolexyz_previous = self.density.calculate_dipole_moment()
            self.itert += 1
        else:
            self.iterpredcop = 0
            self.dipolexyz = (self.density.calculate_dipole_moment()
                              - self.dipolexyz_previous) / self.deltat

        if self.environment == 0 and self.polarization_cavity == [1, 1, 1]:
            # 3D emission (factor 2 for correct WW-emission included)
            # currently the rr_quantization_plane is overloaded with
            # the harmonic frequency [input in eV]
            # rr_amplify is an artificial amplification
            rr_amplify = 1e0
            rr_argument = ((-4.0 * ((self.rr_quantization_plane * Bohr**2)**2
                                    / Hartree**2) * alpha**3 / 3.0
                                 * np.sum(np.square(self.dipolexyz))**0.5)
                           * rr_amplify)
            # function uses V/Angstroem and therefore conversion necessary,
            # it also normalizes the direction which we want to counter
            if np.sum(np.square(self.dipolexyz))**0.5 > 0:
                ext = ConstantElectricField(rr_argument * Hartree / Bohr,
                                            self.dipolexyz)
            else:
                ext = ConstantElectricField(0, [1, 0, 0])
        else:
            # function uses V/Angstroem and therefore conversion necessary
            ext = ConstantElectricField(Hartree / Bohr,
                                        self.polarization_cavity)
        uvalue = 0
        self.ext_i = []
        self.ext_i.append(ext)
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        self.V_iuMM = []
        for ext in self.ext_i:
            V_uMM = []
            hamiltonian = KickHamiltonian(self.hamiltonian, self.density, ext)
            for kpt in self.wfs.kpt_u:
                V_MM = get_matrix(hamiltonian, self.wfs, kpt,
                                  add_kinetic=False, root=-1)
                V_uMM.append(V_MM)
            self.V_iuMM.append(V_uMM)
        self.Ni = len(self.ext_i)

        if self.environment == 0 and self.polarization_cavity != [1, 1, 1]:
            rr_argument = (-4.0 * np.pi * alpha / self.rr_quantization_plane
                           * np.dot(self.polarization_cavity, self.dipolexyz))
        elif self.environment == 0 and self.polarization_cavity == [1, 1, 1]:
            rr_argument = 1.
        elif self.environment == 1:
            if time > 0:
                rr_argument = self.selffield(self.deltat)
            else:
                rr_argument = 0

        Vrr_MM = rr_argument * self.V_iuMM[0][uvalue]
        for i in range(1, self.Ni):
            Vrr_MM += rr_argument * self.V_iuMM[i][uvalue]
        return Vrr_MM

    def dyadicGt(self, deltat, maxtimesteps):
        timeg = np.arange(0, self.deltat *
                          (maxtimesteps * self.frequ_resolution_ampl + 1),
                          self.deltat)
        omegafft = 2 * np.pi * np.fft.fftfreq(len(timeg), self.deltat)
        g_omega = 0
        for omegares in self.cavity_resonance:
            g_omega += (2. / self.cavity_volume / alpha**2 *
                        np.fft.fft(np.exp(-self.cavity_loss * timeg)
                                   * np.sin(omegares * timeg) / omegares))

        if self.ensemble_number is not None:
            if (self.ensemble_loss == 0 or
                self.ensemble_plasmafrequency == 0 or
                self.ensemble_resonance == 0):
                print('Ab initio embedding using dm_ensemblebare_{x/y/z}.dat')
                kickx = read_td_file_kicks('dm_ensemblebare_x.dat')
                kickstrengthx = np.sqrt(np.sum(kickx[0]['strength_v']**2))
                kicky = read_td_file_kicks('dm_ensemblebare_y.dat')
                kickstrengthy = np.sqrt(np.sum(kicky[0]['strength_v']**2))
                kickz = read_td_file_kicks('dm_ensemblebare_z.dat')
                kickstrengthz = np.sqrt(np.sum(kickz[0]['strength_v']**2))
                impdip_x = np.loadtxt('dm_ensemblebare_x.dat', skiprows=5)
                impdip_y = np.loadtxt('dm_ensemblebare_y.dat', skiprows=5)
                impdip_z = np.loadtxt('dm_ensemblebare_z.dat', skiprows=5)
                timedm = impdip_z[:, 0]
                dm_x = impdip_x[:, 2:] / (2. * np.pi * kickstrengthx)
                dm_y = impdip_y[:, 2:] / (2. * np.pi * kickstrengthy)
                dm_z = impdip_z[:, 2:] / (2. * np.pi * kickstrengthz)
                omegafftdm = 2. * np.pi * np.fft.fftfreq(len(timedm),
                                                         timedm[1] - timedm[0])
                # One could move the calculation of the polarizability matrix
                # into some tool and let the user specify directly the location
                # of that matrix. This would make it necessary to have a tool
                # for alpha and the user would need to run it. However, this
                # would also allow to get alpha_ij from another code.
                # Maybe the best option would be to provide both input
                # alternatives, careful with the FFT norms.
                polarizablity_matrix = np.array([[np.fft.fft(dm_x[:, 0]),
                                                  np.fft.fft(dm_y[:, 0]),
                                                  np.fft.fft(dm_z[:, 0])],
                                                 [np.fft.fft(dm_x[:, 1]),
                                                  np.fft.fft(dm_y[:, 1]),
                                                  np.fft.fft(dm_z[:, 1])],
                                                 [np.fft.fft(dm_x[:, 2]),
                                                  np.fft.fft(dm_y[:, 2]),
                                                  np.fft.fft(dm_z[:, 2])]],
                                                dtype=complex)
                pol_pj = np.zeros(len(omegafftdm))
                for ii in range(3):
                    for jj in range(3):
                        pol_pj = pol_pj + (self.polarization_cavity[ii]
                                           * polarizablity_matrix[ii, jj, :]
                                           * self.polarization_cavity[jj])
                pol_interp = interpolate.interp1d(omegafftdm,
                                                  pol_pj,
                                                  kind="cubic",
                                                  fill_value="extrapolate")
                chi_omega = (4. * np.pi * pol_interp(omegafft)
                             * deltat / (timedm[1] - timedm[0]))
            else:
                print('Drude-Lorentz model for polarizablity')
                chi_omega = (self.ensemble_plasmafrequency**2
                             / (self.ensemble_resonance**2 - omegafft**2
                                + 1j * self.ensemble_loss * omegafft))
            G_omega = np.reciprocal(np.reciprocal(g_omega)
                                    - (alpha**2 * omegafft**2 *
                                       self.ensemble_number * chi_omega))
        else:
            G_omega = g_omega
        # alternative: dyadic = np.fft.ifft( 1j * omegafft * G_omega )
        # but at t=0 strong overtones, version below better
        dyadic = -np.gradient(np.fft.ifft(G_omega), deltat)
        return dyadic[:maxtimesteps]

    def selffield(self, deltat):
        # While this is not the most efficient way to write the convolution,
        # it is not the bottleneck of the calculation. If the number of
        # time-steps is largely increased it would be reasonable to use
        # advanced integration strategies in order to shorten the time
        # for convolutions
        dyadic_t = self.dyadic[:self.itert]
        self.dipole_projected[self.itert] = np.dot(self.polarization_cavity,
                                                   self.dipolexyz)
        electric_rr_field = (4. * np.pi * alpha**2 * deltat *
                             (np.dot(dyadic_t[::-1],
                                     self.dipole_projected[:self.itert])
                              - 0.5 * dyadic_t[-1] * self.dipole_projected[0]
                              - (0.5 * dyadic_t[0]
                                 * self.dipole_projected[self.itert])))
        return electric_rr_field
