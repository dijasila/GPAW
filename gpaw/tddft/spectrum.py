from __future__ import print_function
import sys
import numpy as np

from ase.units import Hartree
from gpaw import __version__ as version
from gpaw.mpi import world
from gpaw.tddft.units import (attosec_to_autime, autime_to_attosec,
                              eV_to_aufrequency, aufrequency_to_eV)


# Function for calculating photoabsorption spectrum
def photoabsorption_spectrum(dipole_moment_file, spectrum_file,
                             folding='Gauss', width=0.2123,
                             e_min=0.0, e_max=30.0, delta_e=0.05):
    """Calculates photoabsorption spectrum from the time-dependent
    dipole moment.

    Parameters:

    dipole_moment_file: string
        Name of the time-dependent dipole moment file from which
        the specturm is calculated
    spectrum_file: string
        Name of the spectrum file
    folding: 'Gauss' or 'Lorentz'
        Whether to use Gaussian or Lorentzian folding
    width: float
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lonrentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maxiumum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)


    """


#    kick_strength: [float, float, float]
#        Strength of the kick, e.g., [0.0, 0.0, 1e-3]
#    fwhm: float
#        Full width at half maximum for peaks in eV
#    delta_omega: float
#        Energy resolution in eV
#    max_energy: float
#        Maximum excitation energy in eV

    if folding != 'Gauss':
        raise RuntimeError('Error in photoabsorption_spectrum: '
                           'Only Gaussian folding is currently supported.')

    if world.rank == 0:
        print('Calculating photoabsorption spectrum from file "%s"'
              % dipole_moment_file)

        f_file = open(spectrum_file, 'w')
        dm_file = open(dipole_moment_file, 'r')
        lines = dm_file.readlines()
        dm_file.close()

        for line in lines[:2]:
            assert line.startswith('#')

        # Read kick strength
        columns = lines[0].split('[')
        columns = columns[1].split(']')
        columns = columns[0].split(',')
        kick_strength = np.array([float(columns[0]),
                                  float(columns[1]),
                                  float(columns[2])],
                                 dtype=float)
        strength = np.array(kick_strength, dtype=float)

        print('Using kick strength = ', strength)
        # Continue with dipole moment data
        lines = lines[2:]
        n = len(lines)
        dm = np.zeros((n, 3), dtype=float)
        time = np.zeros((n,), dtype=float)
        for i in range(n):
            data = lines[i].split()
            time[i] = float(data[0])
            dm[i, 0] = float(data[2])
            dm[i, 1] = float(data[3])
            dm[i, 2] = float(data[4])

        t = time - time[0]
        dt = time[1] - time[0]
        dm[:] = dm - dm[0]
        nw = int(e_max / delta_e)
        dw = delta_e * eV_to_aufrequency
        # f(w) = Nw exp(-w^2/2sigma^2)
        # sigma = fwhm / Hartree / (2.* np.sqrt(2.* np.log(2.0)))
        # f(t) = Nt exp(-t^2*sigma^2/2)
        sigma = width * eV_to_aufrequency
        fwhm = sigma * (2. *  np.sqrt(2. * np.log(2.0)))
        kick_magnitude = np.sum(strength**2)

        # write comment line
        f_file.write('# Photoabsorption spectrum from real-time propagation\n')
        f_file.write('# GPAW version: ' + version + '\n')
        f_file.write('# Total time = %lf fs, Time step = %lf as\n' \
            % (n * dt * autime_to_attosec/1000.0, \
               dt * autime_to_attosec))
        f_file.write('# Kick = [%lf,%lf,%lf]\n' % (kick_strength[0], \
                                                   kick_strength[1], \
                                                   kick_strength[2]))
        f_file.write('# %sian folding, Width = %lf eV = %lf Hartree <=> ' \
                     'FWHM = %lf eV\n' % (folding, sigma*aufrequency_to_eV, \
                                          sigma, fwhm*aufrequency_to_eV))

        f_file.write('#  om (eV) %14s%20s%20s\n' % ('Sx', 'Sy', 'Sz'))
        # alpha = 2/(2*pi) / eps int dt sin(omega t) exp(-t^2*sigma^2/2)
        #                                * ( dm(t) - dm(0) )
        # alpha = 0
        for i in range(nw):
            w = i * dw
            # x
            alphax = np.sum(np.sin(t * w) * np.exp(-t**2*sigma**2/2.0) * dm[:,0])
            alphax *= 2 * dt / (2 * np.pi) / kick_magnitude * strength[0]
            # y
            alphay = np.sum(np.sin(t * w) * np.exp(-t**2*sigma**2/2.0) * dm[:,1])
            alphay *= 2 * dt / (2 * np.pi) / kick_magnitude * strength[1]
            # z
            alphaz = np.sum(np.sin(t * w) * np.exp(-t**2*sigma**2/2.0) * dm[:,2])
            alphaz *= 2 * dt / (2 * np.pi) / kick_magnitude * strength[2]

            # f = 2 * omega * alpha
            line = '%10.6lf %20.10le %20.10le %20.10le\n' \
                % (w * aufrequency_to_eV,
                   2 * w * alphax / Hartree,
                   2 * w * alphay / Hartree,
                   2 * w * alphaz / Hartree)
            f_file.write(line)

            if (i % 100) == 0:
                print('.', end=' ')
                sys.stdout.flush()

        print("Sinc contamination", np.exp(-t[-1]**2*sigma**2/2.0))

        print('')
        f_file.close()

        print('Calculated photoabsorption spectrum saved to file "%s"'
              % spectrum_file)

    # Make static method
    # photoabsorption_spectrum=staticmethod(photoabsorption_spectrum)
