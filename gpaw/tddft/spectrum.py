from __future__ import print_function
import numpy as np

from gpaw import __version__ as version
from gpaw.mpi import world
from gpaw.tddft.units import (au_to_as, au_to_fs, au_to_eV)
from gpaw.tddft.folding import FoldedFrequencies
from gpaw.tddft.folding import Folding


def read_dipole_moment_file(fname, remove_duplicates=True):
    # Read kick
    f = file(fname, 'r')
    line = f.readline()
    f.close()
    kick_str_v = line.split('[', 1)[1].split(']', 1)[0].split(',')
    kick_v = np.array(map(float, kick_str_v))

    # Read data
    data_tj = np.loadtxt(fname)
    time_t = data_tj[:, 0]
    norm_t = data_tj[:, 1]
    dm_tv = data_tj[:, 2:]

    # Remove duplicates due to abruptly stopped and restarted calculation
    if remove_duplicates:
        flt_t = np.ones_like(time_t, dtype=bool)
        maxtime = time_t[0]
        for t in range(1, len(time_t)):
            if time_t[t] > maxtime:
                maxtime = time_t[t]
            else:
                flt_t[t] = False

        time_t = time_t[flt_t]
        norm_t = norm_t[flt_t]
        dm_tv = dm_tv[flt_t]

        ndup = len(flt_t) - flt_t.sum()
        if ndup > 0:
            print('Removed %d duplicates' % ndup)

    return kick_v, time_t, norm_t, dm_tv


def calculate_polarizability(data, foldedfrequencies):
    kick_v, time_t, dm_tv = data
    ff = foldedfrequencies
    omega_w = ff.frequencies
    envelope = ff.folding.envelope

    time_t = time_t - time_t[0]
    dt_t = np.insert(time_t[1:] - time_t[:-1], 0, 0.0)
    dm_tv = dm_tv[:] - dm_tv[0]
     
    kick_magnitude = np.sum(kick_v**2)

    Nw = len(omega_w)
    alpha_wv = np.zeros((Nw, 3), dtype=complex)
    f_wt = np.exp(1.0j * np.outer(omega_w, time_t))
    dm_vt = np.swapaxes(dm_tv, 0, 1)
    alpha_wv = np.tensordot(f_wt, dt_t * envelope(time_t) * dm_vt, axes=(1, 1))
    alpha_wv *= kick_v / kick_magnitude
    return alpha_wv


def calculate_photoabsorption(data, foldedfrequencies):
    omega_w = foldedfrequencies.frequencies
    alpha_wv = calculate_polarizability(data, foldedfrequencies)
    abs_wv = 2 / np.pi * omega_w[:, np.newaxis] * alpha_wv.imag
    return abs_wv


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
        Lorentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maximum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)
    """

    if world.rank == 0:
        print('Calculating photoabsorption spectrum from file "%s"'
              % dipole_moment_file)

        r = read_dipole_moment_file(dipole_moment_file)
        kick_v, time_t, norm_t, dm_tv = r

        def str_list(v_i, fmt='%g'):
            return '[%s]' % ', '.join(map(lambda v: fmt % v, v_i))

        print('Using kick strength = %s' % str_list(kick_v))

        freqs = np.arange(e_min, e_max + 0.5 * delta_e, delta_e)
        folding = Folding(folding, width)
        ff = FoldedFrequencies(freqs, folding)
        omega_w = ff.frequencies
        spec_wv = calculate_photoabsorption((kick_v, time_t, dm_tv), ff)
        dt_t = time_t[1:] - time_t[:-1]

        # write comment line
        with open(spectrum_file, 'w') as f:
            def w(s):
                f.write('%s\n' % s)

            w('# Photoabsorption spectrum from real-time propagation')
            w('# GPAW version: %s' % version)
            w('# Total time = %.4f fs, Time steps = %s as' %
              (dt_t.sum() * au_to_fs,
               str_list(np.unique(np.around(dt_t, 6)) * au_to_as, '%.4f')))
            w('# Kick = %s' % str_list(kick_v))
            w('# %sian folding, Width = %.4f eV = %lf Hartree'
              ' <=> FWHM = %lf eV' %
              (folding.folding, folding.width * au_to_eV, folding.width,
               folding.fwhm * au_to_eV))
            w('# %10s %20s %20s %20s' % ('om (eV)', 'Sx', 'Sy', 'Sz'))

            data_wi = np.hstack((omega_w[:, np.newaxis] * au_to_eV,
                                 spec_wv / au_to_eV))
            np.savetxt(f, data_wi, fmt='%12.6lf %20.10le %20.10le %20.10le')

        print('Sinc contamination %.8f' % folding.envelope(time_t[-1]))
        print('Calculated photoabsorption spectrum saved to file "%s"'
              % spectrum_file)
