import numpy as np
from gpaw.external import ConstantElectricField
from ase.units import alpha, Hartree, Bohr
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from scipy import interpolate, special
from gpaw.tddft.spectrum import read_td_file_kicks
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt

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
        [2] loss of cavity,
        [3] (must be string or will be ignored)
            provide a file with the precomputed G(omega), omega > 0,
            the first column should be frequency and the 2rd, 3th and 4th
            should be xx, yy, zz component. This will be generalized
            at a later point.
        [4] using Claussius-Mossoti if 1, otherwise dilute limit.
        [5] increase frequency resolution
        comment: [3] If you provide a file, e.g. using the separate G_dyadic
                     tool, all remaining parameters will be ignored. Notice,
                     that rr_quantization_plane will be used then as
                     artifical amplification, i.e., the internally used G
                     will be G = rr_quantization_plane * G_in. This is done
                     to allow simple parameter scalings but naturally
                     deviates from the ab initio idea.
    environmentens_in: array
        [0] number of ensemble oscillators,
        [1] Ve/V ratio of occupied ensemble to cavity volume,
        [2] resonance frequency of Drude-Lorentz for ensemble,
        [3] lossyness of ensemble oscillators,
        [4] plasma-frequency of ensemble oscillators
        comment: If [2], [3] or [4] equal 0, the code will attempt to load
                 3 dipole files to compute the polarizability matrix.
                 In that case [1] is ignored.
    """

    def __init__(self, rr_qplane_in, pol_cavity_in,
                 environmentcavity_in=None, environmentens_in=None):
        self.rr_quantization_plane = rr_qplane_in / Bohr**2
        self.polarization_cavity = pol_cavity_in
        self.dipolexyz = None
        self.itert = 0
        self.krondelta = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.Ggbamp = 1.
        self.precomputedG = None
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
            self.cutoff_lower = None
            self.cutofffrequency = None
            self.claussius = environmentcavity_in[4]
            if isinstance(environmentcavity_in[3], type('inappropriate')):
                self.precomputedG = environmentcavity_in[3]
                self.frequ_resolution_ampl = 2 * environmentcavity_in[5]
            else:
                self.precomputedG = None
                self.frequ_resolution_ampl = environmentcavity_in[5]
            if environmentens_in is not None:
                self.environmentensemble = 1
                self.ensemble_number = environmentens_in[0]
                self.ensemble_occ_ratio = environmentens_in[1]
                self.ensemble_resonance = (environmentens_in[2]
                                           / Hartree)
                self.dmpval = environmentens_in[3]
                self.ensemble_loss = (environmentens_in[3]
                                      * self.ensemble_resonance)
                self.ensemble_omegap = (environmentens_in[4]
                                        / Hartree)
            else:
                self.ensemble_number = environmentens_in
                self.environmentensemble = 0

    def write(self, writer):
        writer.write(itert=self.itert)
        writer.write(DelDipole=self.dipolexyz * Bohr)
        writer.write(DelDipole_time=self.dipolexyz_time[:self.itert, :] * Bohr)
        writer.write(Dipole_time=self.dipole_time * Bohr)
        writer.write(dipole_projected=self.dipole_projected[:self.itert]
                     * Bohr)
        writer.write(rr_qplane_in=self.rr_quantization_plane
                     * Bohr**2)
        writer.write(pol_cavity_in=self.polarization_cavity)
        if self.environment == 0:
            writer.write(environmentcavity_in=None)
        else:
            writer.write(environmentcavity_in=[self.cavity_resonance[0]
                                               * Hartree,
                                               self.cavity_resonance[-1]
                                               * Hartree,
                                               (self.cavity_loss /
                                                self.cavity_resonance[0]),
                                               self.precomputedG,
                                               self.claussius,
                                               self.frequ_resolution_ampl])
            if self.environmentensemble == 0:
                writer.write(environmentens_in=None)
            else:
                if self.ensemble_resonance > 0:
                    loss_out = self.ensemble_loss / self.ensemble_resonance
                else:
                    loss_out = 0
                writer.write(environmentens_in=[self.ensemble_number,
                                                self.ensemble_occ_ratio,
                                                self.ensemble_resonance,
                                                loss_out,
                                                (self.ensemble_omegap
                                                 * Hartree)])

    def read(self, reader):
        self.itert = reader.itert
        self.dipolexyz = reader.DelDipole / Bohr
        self.dipolexyz_time = reader.DelDipole_time / Bohr
        self.dipole_projected = reader.dipole_projected / Bohr
        self.dipole_time = reader.Dipole_time / Bohr

    def initialize(self, paw):
        if self.dipolexyz is None:
            self.dipolexyz = [0, 0, 0]
            self.dipolexyz_time = [0, 0, 0]
            self.dipole_time = np.zeros((1,3))
            self.dipder3 = np.zeros((1,3))
        self.density = paw.density
        self.wfs = paw.wfs
        self.hamiltonian = paw.hamiltonian
        self.dipolexyz_previous = self.density.calculate_dipole_moment()
        if self.environment == 1:
            self.dyadic = None

    def savelast(self, PC_dip):
        self.itert += 1
        self.dipolexyz_previous = PC_dip
        self.dipole_time = np.vstack((self.dipole_time,
                                      self.dipolexyz_previous))

    def vradiationreaction(self, kpt, time):
        if self.environment == 1 and self.dyadic is None:
            [self.dyadic, self.dyadic_st] = self.dyadicGt(self.deltat,
                                                          self.maxtimesteps)
            static = True
            if static == False:
                self.dyadic_st = self.dyadic_st * 0
                print("You remove the static component from the dyadic.")
            if not hasattr(self, 'dipole_projected'):
                self.dipole_projected = np.zeros(self.maxtimesteps)
                self.dipolexyz_time = np.zeros((self.maxtimesteps, 3))
                self.dipole_time = np.zeros((1,3))
            else:
                self.dipole_projected = \
                    np.concatenate([self.dipole_projected,
                                    np.zeros(self.maxtimesteps)])
                self.dipolexyz_time = \
                    np.concatenate([self.dipolexyz_time,
                                    np.zeros((self.maxtimesteps, 3))])

        # Calculate derivatives
        self.dipolexyz = (self.density.calculate_dipole_moment()
                          - self.dipolexyz_previous) / self.deltat
        if self.environment == 0 and self.polarization_cavity == [1, 1, 1]:
            ### 2nd order version
            # if len(self.dipole_time[:,0]) > 4:
            #     self.dipder3 = ((3/2*self.dipole_time[-4,:]
            #                      - 7*self.dipole_time[-3,:]
            #                      + 12*self.dipole_time[-2,:]
            #                      - 9*self.dipole_time[-1,:]
            #                      + 5/2*self.density.calculate_dipole_moment())
            #                     / self.deltat**3)
            # elif len(self.dipole_time[:,0]) == 4:
            #     self.dipder3 = ((3/2*self.dipole_time[-3,:]
            #                      - 7*self.dipole_time[-3,:]
            #                      + 12*self.dipole_time[-2,:]
            #                      - 9*self.dipole_time[-1,:]
            #                      + 5/2*self.density.calculate_dipole_moment())
            #                     / self.deltat**3)
            # elif len(self.dipole_time[:,0]) == 3:
            #     self.dipder3 = ((3/2*self.dipole_time[-2,:]
            #                      - 7*self.dipole_time[-2,:]
            #                      + 12*self.dipole_time[-2,:]
            #                      - 9*self.dipole_time[-1,:]
            #                      + 5/2*self.density.calculate_dipole_moment())
            #                     / self.deltat**3)
            # elif len(self.dipole_time[:,0]) == 2:
            #     self.dipder3 = ((3/2*self.dipole_time[-1,:]
            #                      - 7*self.dipole_time[-1,:]
            #                      + 12*self.dipole_time[-1,:]
            #                      - 9*self.dipole_time[-1,:]
            #                      + 5/2*self.density.calculate_dipole_moment())
            #                     / self.deltat**3)
            # elif len(self.dipole_time[:,0]) == 1:
            #     self.dipder3 = ((3/2*self.dipole_time[0,:]
            #                      - 7*self.dipole_time[0,:]
            #                      + 12*self.dipole_time[0,:]
            #                      - 9*self.dipole_time[0,:]
            #                      + 5/2*self.density.calculate_dipole_moment())
            #                     / self.deltat**3)*0
            # #self.dipder3[0] = 0
            # #self.dipder3[1] = 0
            # print(self.dipder3)
            ### 1st order version
            if len(self.dipole_time[:,0]) > 2:
                self.dipder3 = ((-self.dipole_time[-3,:]
                                 + 3*self.dipole_time[-2,:]
                                 - 3*self.dipole_time[-1,:]
                                 + self.density.calculate_dipole_moment())
                                / self.deltat**3)
            else:
                self.dipder3 = ((-self.dipole_time[0,:]
                                 - 3*self.dipole_time[-1,:]
                                 + self.density.calculate_dipole_moment())
                                / self.deltat**3)
                if len(self.dipole_time[:,0]) > 1:
                    self.dipder3 += 3*self.dipole_time[-2,:] / self.deltat**3
                else:
                    self.dipder3 += 3*self.dipole_time[0,:] / self.deltat**3

        rr_bg = 0
        if self.environment == 0 and self.polarization_cavity == [1, 1, 1]:
            # 3D emission (factor 2 for correct WW-emission included)
            # currently the rr_quantization_plane is overloaded with
            # the harmonic frequency [input in eV]
            # rr_amplify is an artificial amplification
            rr_amplify = 1e0
            if self.rr_quantization_plane > 0:
                rr_argument_in = (((self.rr_quantization_plane * Bohr**2)**2
                                   / Hartree**2)
                                  * np.sum(np.square(self.dipolexyz))**0.5)
            else:
                rr_argument_in = np.sum(np.square(self.dipder3))**0.5
            rr_argument = -4.0 * alpha**3 / 3.0 * rr_argument_in * rr_amplify
            # function uses V/Angstroem and therefore conversion necessary,
            # it also normalizes the direction which we want to counter
            if np.sum(np.square(self.dipolexyz))**0.5 > 0:
                ext = [ConstantElectricField(rr_argument * Hartree / Bohr,
                                             self.dipolexyz)]
            else:
                ext = [ConstantElectricField(0, [1, 0, 0])]
        elif self.precomputedG is not None:
            ext_x = ConstantElectricField(Hartree / Bohr, [1, 0, 0])
            ext_y = ConstantElectricField(Hartree / Bohr, [0, 1, 0])
            ext_z = ConstantElectricField(Hartree / Bohr, [0, 0, 1])
            ext = [ext_x, ext_y, ext_z]
            """
            rr_bg is the background radiation and uses the provided
            characteristic frequency self.cavity_resonance
            """
            rr_bg = (-4.0 * ((self.cavity_resonance * Bohr**2)**2
                             / Hartree**2) * alpha**3 / 3.0 * self.Ggbamp
                     * np.sum(np.square(self.dipolexyz))**0.5)
        else:
            # function uses V/Angstroem and therefore conversion necessary
            ext = [ConstantElectricField(Hartree / Bohr,
                                         self.polarization_cavity)]
        uvalue = 0
        self.ext_i = ext
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
            rr_argument = [(-4.0 * np.pi * alpha / self.rr_quantization_plane
                            * np.dot(self.polarization_cavity,
                                     self.dipolexyz))]
        elif self.environment == 0 and self.polarization_cavity == [1, 1, 1]:
            rr_argument = [1.]
        elif self.environment == 1:
            if time > 0:
                rr_argument = self.selffield(self.deltat) + rr_bg
                if self.precomputedG is None:
                    rr_argument = [rr_argument.dot(self.polarization_cavity)]
            else:
                rr_argument = [0, 0, 0]

        Vrr_MM = rr_argument[0] * self.V_iuMM[0][uvalue]
        for i in range(1, self.Ni):
            Vrr_MM += rr_argument[i] * self.V_iuMM[i][uvalue]
        return Vrr_MM

    def inverse_quadratic_fit(self, x, y):
        def inv_quad(x, a):
            return a / (x**2)
        popt, pcov = curve_fit(inv_quad, x, y)
        y_fit = inv_quad(x, *popt)
        y_subtracted = y - y_fit
        return [popt, y_subtracted]

    def l_fit(self, x, y, cutfreq):
        cutupper = int(np.floor(len(x) / 2))
        xcut = x[:cutupper]
        cutlower = np.argmin((x[:cutupper] - cutfreq * 1.5)**2)
        xcut = xcut[cutlower:]
        ycut = y[cutlower:cutupper]

        def linear(xcut, a):
            return a * xcut

        popt, pcov = curve_fit(linear, xcut, ycut)
        y_subtracted = y - popt * x
        return [popt, y_subtracted]

    def is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    def dyadicGt(self, deltat, maxtimesteps):
        if os.path.isfile('dyadicD.npz') and maxtimesteps > 1000:
            dyadicD = np.load('dyadicD.npz')
            Dt = dyadicD['Dt']
            Gst = dyadicD['Gst']
            if maxtimesteps <= len(Dt[:, 0]):
                print("You are using the previous Dyadic dyadicD.npz")
                return Dt[:maxtimesteps, :], Gst

        if self.itert > 0:
            self.frequ_resolution_ampl = (float(self.frequ_resolution_ampl) *
                                          float(self.itert) /
                                          float(maxtimesteps - 1))
        timeg = np.arange(0, deltat *
                          (maxtimesteps * self.frequ_resolution_ampl + 1),
                          deltat)
        omegafft = 2 * np.pi * np.fft.fftfreq(len(timeg), deltat)
        Gw0 = np.zeros((len(omegafft), 9), dtype=complex)
        Gst = np.zeros((9, ), dtype=complex)
        Dt = np.zeros((len(omegafft), 9), dtype=complex)
        window_Gw0 = 1.
        window_Gw0_L = 1.
        if self.precomputedG is None:
            g_omega = 0
            for omegares in self.cavity_resonance:
                g_omega += (2. / self.cavity_volume / alpha**2 *
                            np.fft.fft(np.exp(-self.cavity_loss * timeg)
                                       * np.sin(omegares * timeg) / omegares))
            for ii in range(3):
                for jj in range(3):
                    Gw0[:, 3 * ii + jj] = (g_omega *
                                           self.polarization_cavity[ii] *
                                           self.polarization_cavity[jj])
        else:
            G_clean = np.loadtxt(self.precomputedG + '_cleaned.dat',
                                 dtype=complex)
            Gstin = np.loadtxt(self.precomputedG + '_static.dat',
                               dtype=complex)
            flip_xz_cav = True
            if flip_xz_cav is True:
                print("Careful, you are flipping x and z for G!")
                G_clean[:, [1, 9]] = G_clean[:, [9, 1]]
                Gstin[[1, 9]] = Gstin[[9, 1]]
            Gst = Gstin[1:]
            print("Currently, I ignore the static G when dressing with",
                  "molecular polarizabilities. This is not quite",
                  "consistent and should get corrected.")
            for ii in range(len(G_clean[0, 1:])):
                """
                The G that was computed for only some positive frequencies
                G_clean[:,0], is inverted for negative frequencies, attached
                in the np.fft format (0 to max frequency, min frequency to 0)
                and interpolate (GG_pj) to the used frequency resolution as
                specified by the propagation time and
                self.frequ_resolution_ampl. Note, that one has to rescale by
                the frequency-ratio between the oriG_cleanal and
                the interpolated, otherwise the norms are not preserved.
                Recall also that G is hermitian, i.e., G(-w*) = G*(w).
                """
                romin = G_clean[::-1, 0]
                revG = G_clean[::-1, ii + 1]
                omegafullin = np.real(np.append(G_clean[:, 0], -romin[:-1]))
                GG_pj = interpolate.interp1d(omegafullin,
                                             (np.append(G_clean[:, ii + 1],
                                              np.conjugate(revG[:-1]))),
                                             kind="cubic",
                                             fill_value=(0, 0),
                                             bounds_error=False)

                if len(G_clean[0, 1:]) == 3:
                    shiftfac = 4
                elif len(G_clean[0, 1:]) == 9:
                    shiftfac = 1
                else:
                    print('ERROR: G_dyadic.dat is not properly formated')

                """
                The Sellmeier equation can provide a decent model for the
                refractive index over a given frequency domain.
                The window function defined below is adhoc but 'inspired'
                by the idea that high frequencies will not affect the
                valence dynamic of molecules. The function should be inspired
                by the provided G tensor.
                The factor len(omegafft) is supposed to compensate the
                normalization that will happen in the iFFT to get from
                the frequency domain back to time and D(t).
                """
                # self.cutoff_lower = 0.01
                self.cutofffrequency = 0.8
                if self.cutofffrequency is not None:
                    window_Gw0 = 1 - (0.5 + 0.5 *
                                      special.erf((abs(omegafft) -
                                                   self.cutofffrequency) * 25))
                if self.cutoff_lower is not None:
                    window_Gw0_L = (0.5 + 0.5 *
                                    special.erf((abs(omegafft) -
                                                 self.cutoff_lower) * 250))
                    print("CAREFUL, you are using window function(s) for G0w",
                          "and Xw (only upper) with a cutoff-energy [eV] ",
                          str(self.cutofffrequency * Hartree),
                          str(self.cutoff_lower * Hartree))

                    print("CAREFUL, it could be that the cutoffs break",
                          "Kramers-Kroenig and lead to false imaginary",
                          "components in D.")

                Gw0[:, ii * shiftfac] = (GG_pj(omegafft) * window_Gw0 *
                                         window_Gw0_L *
                                         len(omegafft))

        if self.ensemble_number is not None:
            Gw = np.zeros((len(omegafft), 9), dtype=complex)
            Xw = np.zeros((len(omegafft), 9), dtype=complex)
            if (self.ensemble_loss == 0 or
                self.ensemble_omegap == 0 or
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
                dm_x = ((impdip_x[:, 2:] - impdip_x[0, 2:]) /
                        (2. * np.pi * kickstrengthx))
                dm_y = ((impdip_y[:, 2:] - impdip_y[0, 2:]) /
                        (2. * np.pi * kickstrengthy))
                dm_z = ((impdip_z[:, 2:] - impdip_z[0, 2:]) /
                        (2. * np.pi * kickstrengthz))
                omegafftdm = 2. * np.pi * np.fft.fftfreq(len(timedm),
                                                         timedm[1] - timedm[0])
                # One could move the calculation of the polarizability matrix
                # into some tool and let the user specify directly the location
                # of that matrix. This would make it necessary to have a tool
                # for alpha_ij and the user would need to run it. However, this
                # would also allow to get alpha_ij from another code.
                # Maybe the best option would be to provide both input
                # alternatives, careful with the FFT norms.

                lastsize = np.sum(np.abs(dm_x[-1, :]) + np.abs(dm_y[-1, :]) +
                                  np.abs(dm_z[-1, :]))
                dmp = np.ones(len(dm_x[:, 0]))
                if lastsize > self.dmpval and self.dmpval !=0:
                    # and self.precomputedG is not None:
                    print("Adding additional damping to smoothen Xw to ",
                          self.dmpval)
                    decayrate = -np.log(self.dmpval) / len(dm_x[:, 0])
                    dmp = np.exp(-decayrate * np.arange(len(dm_x[:, 0])))

                pol_matrix = np.array([np.fft.fft(dm_x[:, 0] * dmp),
                                       np.fft.fft(dm_x[:, 1] * dmp),
                                       np.fft.fft(dm_x[:, 2] * dmp),
                                       np.fft.fft(dm_y[:, 0] * dmp),
                                       np.fft.fft(dm_y[:, 1] * dmp),
                                       np.fft.fft(dm_y[:, 2] * dmp),
                                       np.fft.fft(dm_z[:, 0] * dmp),
                                       np.fft.fft(dm_z[:, 1] * dmp),
                                       np.fft.fft(dm_z[:, 2] * dmp)],
                                      dtype=complex)

                print('Embedding RR, particle density is ',
                      self.ensemble_number / self.cavity_volume,
                      'with particle number',
                      self.ensemble_number,
                      'and volume of',
                      self.cavity_volume)
                # alpha_ij is the polarizablity (local response)
                alpha_ij = np.zeros((len(omegafft),9),dtype=complex)
                for ii in range(9):
                    pol_interp = interpolate.interp1d(omegafftdm,
                                                      pol_matrix[ii, :],
                                                      kind="cubic",
                                                      fill_value="extrapolate")
                    alpha_ij[:,ii]=(pol_interp(omegafft)
                                    * deltat / (timedm[1] - timedm[0])
                                    * window_Gw0)
                print("Increase in frequency resolution",
                      len(omegafft) / len(omegafftdm))

                if self.claussius == 1:
                    print('Using Claussius-Mossotti for Xw')
                    for el in range(len(omegafft)):
                        if self.is_invertible(np.reshape(alpha_ij[el, :], (3, 3))):
                            # Xw[el, :] = np.reshape((4. * np.pi * self.ensemble_number / self.cavity_volume) * np.reshape(alpha_ij[el, :], (3, 3)) @ (np.linalg.inv(np.eye(3)) - 1. / 3. * (4. * np.pi * self.ensemble_number / self.cavity_volume) * np.reshape(alpha_ij[el, :], (3, 3))) , (-1, ))
                            Xw[el, :] = np.reshape(np.linalg.inv(np.linalg.inv(np.reshape(alpha_ij[el, :], (3, 3))) / (4. * np.pi * self.ensemble_number / self.cavity_volume) - 1. / 3. * np.eye(3)), (-1, ))
                        else:
                            Xw[el, :] = 4. * np.pi * self.ensemble_number / self.cavity_volume * alpha_ij[el, :]
                            print("Polarizability not invertible! Skipping CM-step")

                for ii in range(9):
                    plt.figure()
                    if self.claussius == 1:
                        plt.plot(omegafft * Hartree, np.real(Xw[:, ii]))
                        plt.plot(omegafft * Hartree, np.imag(Xw[:, ii]))
                    else:
                        plt.plot(omegafft * Hartree,
                                 (4. * np.pi
                                  * self.ensemble_number / self.cavity_volume
                                  * np.real(alpha_ij[:, ii])))
                        plt.plot(omegafft * Hartree,
                                 (4. * np.pi
                                  * self.ensemble_number / self.cavity_volume
                                  * np.imag(alpha_ij[:, ii])))
                    plt.xlabel("Energy (eV)")
                    plt.ylabel(r"$\chi(\omega)$, i=" + str(ii)
                               + 'CM=' + str(self.claussius) )
                    if self.cutofffrequency is not None:
                        plt.xlim(0, self.cutofffrequency * 1.5 * Hartree)
                    else:
                        plt.xlim(0, 14)
                    plt.savefig('Xw_' + str(ii) + '.png')
                    plt.close()
            else:
                print('Drude-Lorentz model for polarizablity')
                for ii in range(3):
                    alpha_ij[:, ii * 4] = (self.ensemble_omegap**2
                                           / (self.ensemble_resonance**2 -
                                              omegafft**2 + (1j * self.ensemble_loss
                                                             * omegafft)))
            if self.precomputedG is None:
                print("Dressing G for simplified cavity")
                for ii in range(3):
                    for jj in range(3):
                        Gw[:, 3 * ii + jj] = (
                            self.polarization_cavity[ii] *
                            self.polarization_cavity[jj] *
                            np.reciprocal(
                                np.reciprocal(g_omega) -
                                (
                                    alpha ** 2 * omegafft ** 2 * 4 * np.pi *
                                    self.ensemble_number * alpha_ij[:, 3 * ii + jj] *
                                    self.polarization_cavity[ii] *
                                    self.polarization_cavity[jj]
                                )
                            )
                        )
                if self.claussius == 1:
                    Gwibar = np.zeros((len(omegafft), 9), dtype=complex)
                    for ii in range(3):
                        for jj in range(3):
                            Gwibar[:, 3 * ii + jj] = (np.reciprocal(g_omega) *
                                                      self.polarization_cavity[ii] *
                                                      self.polarization_cavity[jj])
                    for el in range(len(omegafft)):
                        if not self.is_invertible(np.reshape(Gwibar[el, :], (3, 3))- (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3)))):
                            shift = 1e-8
                        else:
                            shift = 0
                        Gw[el, :] = np.reshape((np.linalg.inv( (np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                                                                (np.reshape(Gwibar[el, :]+shift, (3, 3)) @ (np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3)))))
                                                              - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3))))
                                                @ np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)))
                                                @ (np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)) @ np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3)))) )
                                               ,(-1, ))
                        # Gw[el, :] = np.reshape(np.linalg.inv(np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                        #                        np.reshape(Gw[el, :], (3, 3)) @ (np.eye(3) + 1. / 3. * np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))) @ np.reshape(Xw[el, :], (3, 3)) )
                        #                        ,(-1, ))
            else:
                print("Dressing G using the provided Gw0,",
                      "this may take a few minutes.")
                bg_correction = False
                if bg_correction == False:
                    print("Using new version with local field correction but ignoring free-space dressing.")
                if bg_correction:
                    G_freespace = np.zeros((len(omegafft), 9), dtype=complex)
                    Gbg = np.zeros((len(omegafft), 9), dtype=complex)
                    for ii in range(9):
                        G_freespace[:, ii] = (1j * omegafft * alpha /
                                              (6 * np.pi) * self.krondelta[ii])
                    Gw0 = Gw0 + G_freespace * len(omegafft)
                    """
                    The factor len(omegafft) is again added to compensate the
                    normalization via the FFT.
                    """
                for el in range(len(omegafft)):
                    """
                    For each frequency, reshape Gw0 and Xw into matrix and
                    build Gw via taking the inverse. Notice, that for Xw=0,
                    Gw=Gw0 and the singular case is also set to Gw=Gw0.
                    """
                    if self.claussius == 1:
                        print("dressed G1 expression is wrong here")
                        stop
                        if self.is_invertible(np.reshape(Gw0[el, :], (3, 3))):
                            Gw[el, :] = np.reshape((np.linalg.inv( (np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                                                                    (np.linalg.inv(np.reshape(Gw0[el, :], (3, 3))) @ (np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3)))))
                                                                  - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3))))
                                                    @ np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)))
                                                    @ (np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)) @ np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3)))) )
                                                   ,(-1, ))
                            # Gw[el, :] = np.reshape(np.linalg.inv(np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                            #                        (np.linalg.inv(np.linalg.inv(np.reshape(Gw0[el, :], (3, 3))) - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3)))) @
                            #                         (np.eye(3) + 1. / 3. * np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))) @ np.reshape(Xw[el, :], (3, 3)) )
                            #                         )
                            #                        ,(-1, ))
                        else:
                            Gw[el, :] = Gw0[el, :]
                    else:
                        if self.is_invertible(np.reshape(Gw0[el, :], (3, 3))):
                            Gw[el, :] = np.reshape(
                                np.linalg.inv(
                                    np.linalg.inv(
                                        np.reshape(Gw0[el, :], (3, 3))
                                    ) - (alpha**2 * omegafft[el]**2 *
                                         self.ensemble_number * 4 * np.pi *
                                         np.reshape(alpha_ij[el, :], (3, 3)))
                                ),
                                (-1, )
                            )
                        else:
                            Gw[el, :] = Gw0[el, :]
                if bg_correction == True and self.cutofffrequency is not None:
                    """
                        -- adding local field correction
                                       derived by Frieder
                    print("Setting up Gbar0 correction -- only ISO yet")
                    iso_eps = 1 + (Xw[:, 0] + Xw[:, 4] + Xw[:, 8])/3
                    for ii in range(3):
                        for jj in range(3):
                            Gbar0[:,ii,jj] = (
                                self.krondelta[ii,jj] *
                                1/(3.* V_e * iso_eps * omegafft ** 2)
                    --- missing: define V_e somewhere
                    """
                    for ii in range(3):
                        [Gbg_re, Gout_re] = self.l_fit(omegafft,
                                                       np.real(Gw[:, 4 * ii]),
                                                       self.cutofffrequency)
                        [Gbg_im, Gout_im] = self.l_fit(omegafft,
                                                       np.imag(Gw[:, 4 * ii]),
                                                       self.cutofffrequency)
                        if (np.abs(Gbg_im / len(omegafft)
                            / (alpha / (6 * np.pi)) - 1) > 0.1):
                            Gbg[:, 4 * ii] = 1j * Gbg_im * omegafft
                            self.Ggbamp = (Gbg_im / len(omegafft) /
                                           (alpha / (6 * np.pi)))
                            print('background emission amplification: ',
                                  self.Ggb_amplification)
                        else:
                            Gbg[:, 4 * ii] = (1j * alpha / (6 * np.pi) *
                                              omegafft * len(omegafft))
                            self.Ggbamp = 1.
                        if np.abs(Gbg_re) > 1e6:
                            print("# WARNING: The linear part of G has a",
                                  "sizeable real-part, which is not take into",
                                  "account Gbg_re = ", str(Gbg_re))
                    Gw = Gw - Gbg
                    Gw0 = Gw0 - Gbg
                elif bg_correction == True and self.cutofffrequency is None:
                    Gw = Gw - G_freespace
                    Gw0 = Gw0 - G_freespace
        else:
            Gw = Gw0
            print('USING BARE Gw0')
        if self.precomputedG is not None:
            # We can move the sign to Dt later but the previous version
            # was missing a sign in g and for test-suit reasons I added this
            Gw = Gw * (-1)
            Gw0 = Gw0 * (-1)
            # if self.rr_quantization_plane * Bohr**2 != 1.:
            #     Gw = Gw * self.rr_quantization_plane * Bohr**2
            #     print("CAREFUL, you amplify the strength of G by a factor",
            #           "self.rr_quantization_plane=",
            #           self.rr_quantization_plane * Bohr**2)
        for ii in range(9):
            Dt[:, ii] = -np.gradient(np.fft.ifft(Gw[:, ii].flatten()), deltat)
            # For some reason the explicit derivative works best, version
            # below also possible but seems to have issues sometimes.
            # Notice that I moved a minus from down here to the sine G0
            # because the poles had been defined up side down
            # NOTE - I moved the sign back because there seem to be minor
            # numerical deviations otherwise -- check that again later
            # Dt[:,ii] = np.fft.ifft(1j * omegafft * Gw[:,ii].flatten())
            if ii==8:
                plt.figure()
                plt.plot(omegafft * Hartree, np.real(Gw[:, ii]), 'k-',
                         label='Real Gw-scatter Element: ' + str(ii))
                plt.plot(omegafft * Hartree, np.imag(Gw[:, ii]), 'r-',
                         label='Imag Gw-scatter Element: ' + str(ii))
                plt.plot(omegafft * Hartree, np.real(Gw0[:, ii]), 'k:',
                         label='Real Gw0 Element: ' + str(ii))
                plt.plot(omegafft * Hartree, np.imag(Gw0[:, ii]), 'r:',
                         label='Imag Gw0 Element: ' + str(ii))
                plt.xlabel("Energy (eV)")
                plt.ylabel(r"$G^{(1),no static}_i(\omega)$, i=" + str(ii))
                if self.cutofffrequency is not None:
                    plt.xlim(0, self.cutofffrequency * 1.5 * Hartree)
                else:
                    plt.xlim(0, 14)
                plt.legend(loc="upper right")
                plt.savefig('Gw_' + str(ii) + '.png')
                #plt.close()
                plt.figure()
                plt.plot(range(maxtimesteps), np.real(Dt[:maxtimesteps, ii]),
                         label='Real Dt Element: ' + str(ii))
                plt.plot(range(maxtimesteps), np.imag(Dt[:maxtimesteps, ii]),
                         label='Imag Dt Element: ' + str(ii))
                plt.xlabel("time step")
                plt.ylabel(r"$D^{(1),no static}_i(t)$, i=" + str(ii))
                plt.legend(loc="upper right")
                plt.savefig('Dt_' + str(ii) + '.png')
        np.savez("Xw",
                 energy=omegafft[:int(len(omegafft)/2)] * Hartree,
                 alpha_ij=alpha_ij[:int(len(omegafft)/2), :],
                 Xw=Xw[:int(len(omegafft)/2), :])
        np.savez("dyadicD",
                 Dt=Dt[:int(len(omegafft)/2), :],
                 energy=omegafft[:int(len(omegafft)/2)] * Hartree,
                 Gw=Gw[:int(len(omegafft)/2), :],
                 Gst=Gst)
        return [Dt[:maxtimesteps, :], Gst]

    def selffield(self, deltat):
        # While this is not the most efficient way to write the convolution,
        # it is not the bottleneck of the calculation. If the number of
        # time-steps is largely increased it would be reasonable to use
        # advanced integration strategies in order to shorten the time
        # for convolutions
        self.dipole_projected[self.itert] = np.dot(self.polarization_cavity,
                                                   self.dipolexyz)
        self.dipolexyz_time[self.itert, :] = self.dipolexyz
        electric_rr_field = np.zeros((3, ), dtype=complex)
        if self.itert == 0:
            RuntimeError('Use the scpc propagator for the RR potential.')
        for ii in range(3):
            for jj in range(3):
                dyadic_t = self.dyadic[:self.itert, 3 * ii + jj]
                electric_rr_field[ii] += (
                    (4. * np.pi * alpha**2 * deltat *
                     (np.dot(dyadic_t[::-1],
                             self.dipolexyz_time[:self.itert, jj])
                      - 0.5 * dyadic_t[-1] * self.dipolexyz_time[0, jj]
                      - (0.5 * dyadic_t[0] *
                         self.dipolexyz_time[self.itert, jj])))
                    + (4. * np.pi * alpha**2 * self.dyadic_st[3 * ii + jj] *
                       self.density.calculate_dipole_moment()[jj]))
        return np.real(electric_rr_field)
