import numpy as np
from gpaw.external import ConstantElectricField
from ase.units import alpha, Hartree, Bohr
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from gpaw.tddft.units import attosec_to_autime
import scipy, os, sys

# remove below
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt


class RRemission(object):
    r"""
    Radiation-reaction potential according to Schaefer et al.
    [arXiv 2109.09839] The potential accounts for the friction
    forces acting on the radiating system of oscillating charges
    emitting into a single dimension. A more elegant
    formulation would use the current instead of the dipole.
    Please contact christian.schaefer.physics@gmail.com if any problems
    should appear or you would like to consider more complex emission.
    Big thanks to Tuomas Rossi and Jakub Fojt for their help.

    Parameters
    ----------
    rr_quantization_plane: float
        value of :math:`rr_quantization_plane` in atomic units
    pol_cavity: array
        value of :math:`pol_cavity` dimensionless (directional)
    environmentcavity_in: array
        [0] lowest harmonic of cavity, [1] highest harmonic (cutoff), [2] loss of cavity,
        [3] timestep used in propagation, [4] number of timesteps, [5] increase frequency resolution
        comment: [3 and 4] are needed as this routine is defined before the propagator is defined,
                 ideally we find a way to avoid that those numbers have to be provided twice
    environmentensemble_in: array
        [0] number of ensemble oscillators, [1] Ve/V ratio of occupied ensemble to cavity volume,
        [2] resonance frequency of Drude-Lorentz for ensemble, [3] lossyness of ensemble oscillators,
        [4] plasma-frequency of ensemble oscillators
        comment: those parameters could be ideally read-in or fitted from a previous spectrum
    """

    def __init__(self, rr_quantization_plane_in, pol_cavity_in, environmentcavity_in=None, environmentensemble_in=None):
        self.rr_quantization_plane = rr_quantization_plane_in / Bohr**2
        self.polarization_cavity = pol_cavity_in
        self.dipolexyz = None
        if environmentcavity_in is None:
          self.environment = 0
        else:
          self.environment = 1
          self.cavity_resonance = np.arange(environmentcavity_in[0], environmentcavity_in[1] + 1e-8, environmentcavity_in[0]) / Hartree
          self.cavity_volume = self.rr_quantization_plane * (np.pi / (alpha * self.cavity_resonance[0])) # 1 / S(x0)S(x0)
          self.cavity_loss = environmentcavity_in[2] * self.cavity_resonance[0]
          self.deltat = environmentcavity_in[3] * attosec_to_autime
          self.maxtimesteps = environmentcavity_in[4] + 1 # the +1 is because time 0 is not included in the time-steps
          self.frequ_resolution_ampl = environmentcavity_in[5]
          if environmentensemble_in is not None:
              self.ensemble_number = environmentensemble_in[0]
              self.ensemble_occupation_ratio = environmentensemble_in[1]
              self.ensemble_resonance = environmentensemble_in[2] / Hartree
              self.ensemble_loss = environmentensemble_in[3] * self.ensemble_resonance
              self.ensemble_plasmafrequency = environmentensemble_in[4] / Hartree
          else:
              self.ensemble_number = environmentensemble_in

    def write(self, writer):
        writer.write(DelDipole=self.dipolexyz)

    def read(self, reader):
        self.dipolexyz = reader.DelDipole

    def initialize(self, paw):
        self.iterpredcop = 0
        self.time_previous = paw.time
        if self.dipolexyz is None: 
            self.dipolexyz = [0, 0, 0]
        self.density = paw.density
        self.wfs = paw.wfs
        self.hamiltonian = paw.hamiltonian
        self.dipolexyz_previous = self.density.calculate_dipole_moment()
        self.timestep = 0
        if self.environment  == 1:
            self.dipole_projected = np.zeros(self.maxtimesteps)
            self.dyadic = self.dyadicGt(self.maxtimesteps)
            self.frequ_resolution_ampl = 1.

    def vradiationreaction(self, kpt, time):
        if self.iterpredcop == 0:
            self.iterpredcop += 1
            self.dipolexyz_previous = self.density.calculate_dipole_moment()
            self.time_previous = time
            deltat = 1
            self.timestep += 1
        else:
            self.iterpredcop = 0
            deltat = time - self.time_previous
            self.dipolexyz = (self.density.calculate_dipole_moment()
                              - self.dipolexyz_previous) / deltat

        if self.environment  == 0 and self.polarization_cavity == [1,1,1]:
            # 3D emission (factor 2 for correct WW-emission included)
            # currently the rr_quantization_plane is overloaded with the harmonic frequency [input in eV]
            rr_amplify = 1e0 # artificial amplification of decay in order to obtain lifetime
            rr_argument = (-4.0 * ((self.rr_quantization_plane * Bohr**2)**2 / Hartree**2) * alpha**3 / 3.0
                          * np.sum(np.square(self.dipolexyz))**0.5) * rr_amplify
            # function uses V/Angstroem and therefore conversion necessary, it also normalizes the direction
            # so we have to counter that effect,
            if np.sum(np.square(self.dipolexyz))**0.5 > 0:
                ext = ConstantElectricField(rr_argument * Hartree / Bohr, self.dipolexyz)
            else:
                ext = ConstantElectricField(0, [1,0,0])
        else:
            # function uses V/Angstroem and therefore conversion necessary
            ext = ConstantElectricField(Hartree / Bohr, self.polarization_cavity)
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

        if self.environment  == 0 and self.polarization_cavity != [1,1,1]:
            rr_argument = (-4.0 * np.pi * alpha / self.rr_quantization_plane
                          * np.dot(self.polarization_cavity, self.dipolexyz))
        elif self.environment  == 0 and self.polarization_cavity == [1,1,1]:
            rr_argument = 1.
        elif self.environment == 1:
            if time > 0:
                rr_argument = self.selffield()
            else:
                rr_argument = 0

        Vrr_MM = rr_argument * self.V_iuMM[0][uvalue]
        for i in range(1, self.Ni):
            Vrr_MM += rr_argument * self.V_iuMM[i][uvalue]
        return Vrr_MM

    def dyadicGt(self, maxtimesteps):
        loadedG = 0
        loadedD = 0
        printGD = 0
        # This version to load the dyadics directly leads to dumping huge files and loading them is not really faster than creating them.
        # It might be better to save/load only an energy-window and grep from the meta-data the full size. Then patch the rest with zeros.
        if self.ensemble_number is not None:
            safenamefiwG = str('dyadicfiwG_'+str(self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1))+'_'+str(self.deltat)+'_'+str(self.cavity_resonance[0])+'_'+str(self.cavity_volume)+'_'+str(self.cavity_loss)+'_'+str(self.frequ_resolution_ampl)+'_'+str(self.ensemble_number)+'_'+str(self.ensemble_occupation_ratio)+'.dat' )
        else:
            safenamefiwG = str('dyadicfiwG_'+str(self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1))+'_'+str(self.deltat)+'_'+str(self.cavity_resonance[0])+'_'+str(self.cavity_volume)+'_'+str(self.cavity_loss)+'_'+str(self.frequ_resolution_ampl)+'.dat' )
        if self.ensemble_number!=0 and os.path.exists(safenamefiwG):
            dyadic = np.loadtxt(safenamefiwG, dtype=np.complex_)
            loadedD = 1
            print('Restart with existing FFT(iwG)(t)')
        safenameGw = str('dyadicGw_'+str(self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1))+'_'+str(self.deltat)+'_'+str(self.cavity_resonance[0])+'_'+str(self.cavity_volume)+'_'+str(self.cavity_loss)+'_'+str(self.frequ_resolution_ampl)+'.dat' )
        if loadedD == 0 and os.path.exists(safenameGw):
            inputG = np.loadtxt(safenameGw, dtype=np.complex_) 
            omegafft = inputG[:,0]
            g_omega = inputG[:,1]
            loadedG = 1
            print('Restart with existing G(omega)')
        if loadedD == 0 and loadedG == 0:
            timeg = np.arange(0, self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1), self.deltat)
            omegafft = 2*np.pi*np.fft.fftfreq(len(timeg),self.deltat)
            g_omega = 0
            for omegares in self.cavity_resonance:
                g_omega += 2. / self.cavity_volume / alpha**2 * np.fft.fft( np.exp(-self.cavity_loss*timeg)*np.sin(omegares*timeg)/omegares )
            if printGD:
                np.savetxt(safenameGw, np.column_stack((omegafft, g_omega)))
            #safenameGw = str('dyadicGw_'+str(self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1))+'_'+str(self.deltat)+'_'+str(self.cavity_resonance[0])+'_'+str(self.cavity_volume)+'_'+str(self.cavity_loss)+'_'+str(self.frequ_resolution_ampl)+'.dat' )
            #safeobject = np.array([self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1), self.deltat, self.cavity_resonance[0],self.cavity_volume,self.cavity_loss,self.frequ_resolution_ampl])
            #with ase.parallel.paropen('dyadicGw.dat', 'ab') as dyadicGw:
            #    np.savetxt(dyadicGw, np.real(g_omega))
            #    np.savetxt(dyadicGw, np.imag(g_omega))
            #dyadicGw = ase.parallel.paropen('dyadicGw.dat', 'w+')
            #ase.parallel.parprint([self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1), self.deltat, self.cavity_resonance[0],self.cavity_volume,self.cavity_loss,self.frequ_resolution_ampl], file=dyadicGw)
            #ase.parallel.parprint(np.real(g_omega), file=dyadicGw)
            #ase.parallel.parprint(np.imag(g_omega), file=dyadicGw)
            #np.savetxt('dyadicGw.dat', np.vstack(( np.array([self.deltat*(maxtimesteps*self.frequ_resolution_ampl+1), self.deltat, self.cavity_resonance,self.cavity_volume,self.cavity_loss,self.frequ_resolution_ampl]), g_omega )) )
        if loadedD == 0:
            if self.ensemble_number is not None:
                if self.ensemble_loss == 0 or self.ensemble_plasmafrequency == 0 or self.ensemble_resonance == 0:
                    isotropic = 0
                    kickstrength = 1e-5
                    print('Ab-initio derived polarizablity using isotropic = '+ str(isotropic) + ' with dm_ensemblebare_{x/y/z}.dat and kick-strength '+str(kickstrength))
                    if isotropic == 1:
                        importedabinitiodipole_x = np.loadtxt('dm_ensemblebare_x.dat',skiprows=5)
                        importedabinitiodipole_y = np.loadtxt('dm_ensemblebare_y.dat',skiprows=5)
                        importedabinitiodipole_z = np.loadtxt('dm_ensemblebare_z.dat',skiprows=5)
                        if (len(importedabinitiodipole_x) != len(importedabinitiodipole_y)) or (len(importedabinitiodipole_x) != len(importedabinitiodipole_z)):
                            print('Error - Please provide the bare ensemble dipole for each kick-direction (xx, yy, zz) with equal length.')
                            sys.exit()
                        timedm = importedabinitiodipole_x[:,0]
                        dm_x = importedabinitiodipole_x[:,2:]
                        dm_y = importedabinitiodipole_y[:,2:]
                        dm_z = importedabinitiodipole_z[:,2:]
                        polarizablity = 1. / 3. * ((np.fft.fft(dm_x[:,0])/kickstrength)
                                                  + (np.fft.fft(dm_y[:,1])/kickstrength)
                                                  + (np.fft.fft(dm_z[:,2])/kickstrength) )  
                    else:
                        importedabinitiodipole_z = np.loadtxt('dm_ensemblebare_z.dat',skiprows=5)
                        timedm = importedabinitiodipole_z[:,0]
                        dm_z = importedabinitiodipole_z[:,2:]
                        polarizablity = np.fft.fft(dm_z[:,2]) / kickstrength
                    omegafftdm = 2. * np.pi*np.fft.fftfreq(len(timedm), timedm[1]-timedm[0])
                    polarizablity_interp = scipy.interpolate.interp1d(omegafftdm, polarizablity, kind="cubic", fill_value="extrapolate")
                    polarizablity_stretched = polarizablity_interp(omegafft)
                    ensemble_volume_factor = 1./(2.*np.pi) # This factor should be 1 if my derivations are correct but 1/2pi is much better. Maybe the FT of the kick is defined with 2pi somewhere internally
                    chi_omega = 4. * np.pi * ensemble_volume_factor * polarizablity_stretched * self.deltat / (timedm[1]-timedm[0])
                    if ensemble_volume_factor != 1.:
                        print('CAREFUL: Using an artificial ensemble-volume-factor of '+str(ensemble_volume_factor)+' for the zz-polarization embedding.')
                else:
                    print('Drude-Lorentz model for polarizablity')
                    chi_omega = self.ensemble_plasmafrequency**2 / ( self.ensemble_resonance**2 - omegafft**2 + 1j * self.ensemble_loss * omegafft )
                G_omega = np.reciprocal(np.reciprocal(g_omega) - alpha**2 * omegafft**2 * self.ensemble_number * chi_omega )
                #G_omega = np.reciprocal(np.reciprocal(g_omega) - alpha**2 * omegafft**2 * self.ensemble_occupation_ratio * self.cavity_volume * self.ensemble_number * chi_omega )
            else:
                G_omega = g_omega
            #dyadicdummy = np.fft.ifft( 1j * omegafft * G_omega ) # at t=0 unstable, leading to strong overtones, derivative version below is preferential
            dyadic = -np.gradient(np.fft.ifft(G_omega),self.deltat)
            if printGD:
                np.savetxt(safenamefiwG, dyadic)
        #plt.plot(np.real(omegafft)*Hartree,np.imag(g_omega),label='Im(g(w))')
        #plt.plot(np.real(omegafft)*Hartree,np.imag(G_omega),'-.',label='Im(G(w))')
        #plt.plot(np.real(omegafftdm)*Hartree,np.imag(polarizablity),'ko',label='raw data')
        #plt.plot(np.real(omegafft)*Hartree,np.imag(polarizablity_stretched),'r--',label='Interpolated')
        #plt.xlim([2.,4.])
        #plt.axvline(x=2.682)
        #plt.axvline(x=3.192)
        #plt.legend()
        #plt.show()
        #stop
        return dyadic[:maxtimesteps]

    def selffield(self):
        # While this is not the most efficient way to write the convolution, it is not the bottleneck of the calculation
        # If the number of time-steps is largely increased it would be reasonable to use advanced integration
        # strategies in order to shorten the time for convolutions
        dyadic_t = self.dyadic[:self.timestep]
        self.dipole_projected[self.timestep] = np.dot(self.polarization_cavity, self.dipolexyz)
        electric_rr_field = ( 4. * np.pi * alpha**2 * self.deltat *
                             (np.dot(dyadic_t[::-1], self.dipole_projected[:self.timestep])
                              - 0.5 * dyadic_t[-1] * self.dipole_projected[0]
                              - 0.5 * dyadic_t[0] * self.dipole_projected[self.timestep]) )
        return electric_rr_field
