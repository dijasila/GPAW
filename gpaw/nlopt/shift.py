
# Import the required modules: General
import numpy as np

# Import the required modules: GPAW/ASE
from ase.units import Bohr, _hbar, _e, _me
from ase.utils.timing import Timer
from gpaw.mpi import world
from gpaw.nlopt.basic import load_data, parprint, print_progressbar
from gpaw.nlopt.nlobas import get_rml, calc_gender

# Compute the SHG spectrum and save it


def get_shift(
        freqs=[1.0],
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        ftol=1e-4, Etol=1e-6,
        band_n=None,
        out_name='shift.npy',
        mml_name='mml.npz'):
    """
    Calculate RPA shift current for nonmagnetic semiconductors

    Input:
        freqs           Excitation frequency array (a numpy array or list)
        eta             Broadening, a number or an array (default 0.05 eV)
        pol             Tensor element (default 'yyy')
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        band_n          List of bands in the sum (default 0 to nb)
        out_name        Output filename (default 'shift.npy')
        mml_name        The momentum filename (default 'mml.npz')
    Output:
        shift.npy       Numpy array containing the spectrum and frequencies
    """

    # Start a timer
    timer = Timer()
    parprint('Calculating shift current (in {:d} cores).'.format(world.size))

    # Useful variables
    pol_v = ['xyz'.index(ii) for ii in pol]
    w_l = np.array(freqs)
    nw = len(freqs)
    parprint('Calculation for element {}.'.format(pol))

    # Load the required data
    with timer('Load and distribute the data'):
        k_info = load_data(mml_name=mml_name)
        k_ind, tmp = k_info.popitem()
        nb = len(tmp[1])
        nk = len(k_info) * world.size  # Approximately
        if band_n is None:
            band_n = list(range(nb))
        est_mem = 6 * 3 * nk * nb**2 * 16 / 2**20
        parprint('At least {:.2f} MB of memory is required.'.format(est_mem))
        # print(k_info.keys())

    # Initial call to print 0% progress
    count = 0
    ncount = len(k_info)
    print_progressbar(count, ncount)

    # Initialize the outputs
    sum2_l = np.zeros((nw), complex)

    # Do the calculations
    for k_ind, (we, f_n, E_n, p_vnn) in k_info.items():
        with timer('Position matrix elements calculation'):
            r_vnn, D_vnn = get_rml(E_n, p_vnn, pol_v, Etol=Etol)

        with timer('Compute generalized derivative'):
            rd_vvnn = calc_gender(E_n, r_vnn, D_vnn, pol_v, Etol=Etol)

        with timer('Sum over bands'):
            tmp = calc_shift(
                eta, w_l, f_n, E_n, r_vnn, rd_vvnn, band_n, pol_v,
                ftol=ftol, Etol=Etol)

        # Add it to previous with a weight
        sum2_l += tmp * we

        # Print the progress
        count += 1
        print_progressbar(count, ncount)

    with timer('Gather data from cores'):
        world.sum(sum2_l)

    # Make the output in SI unit (2 is for spin)
    dim_init = -1j * 2 * _e**3 / (2 * _hbar * (2.0 * np.pi)**3)
    dim_sum = (_hbar / (Bohr * 1e-10))**3 / \
        (_e**4 * (Bohr * 1e-10)**3) * (_hbar / _me)**3
    dim_SI = 1j * dim_init * dim_sum  # 1j due to imag in loop
    sigma_l = dim_SI * sum2_l.real

    # A multi-col output
    shift = np.vstack((freqs, sigma_l))

    # Save it to the file
    if world.rank == 0:
        np.save(out_name, shift)

        # Print the timing
        timer.write()

    # Return
    return shift


def calc_shift(
        eta, w_l, f_n, E_n, r_vnn, rd_vvnn, band_n, pol_v,
        ftol=1e-4, Etol=1e-6):
    """
    Loop over bands for computing in length gauge

    Input:
        eta             Broadening
        w_l             Complex frequency array
        f_n             Fermi levels
        E_n             Energies
        r_vnn           Momentum matrix elements
        rd_vvnn         Generalized derivative of position
        pol_v           Tensor element
        band_n          Band list
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
    Output:
        sum2_l          Output array
    """

    # Initialize variable
    sum2_l = np.zeros((w_l.size), complex)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            # Remove the non important term (use TRS)
            if mmi <= nni:
                continue
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni]

            # Two band part
            if np.abs(fnm) > ftol:
                tmp = np.imag(
                    r_vnn[pol_v[1], mmi, nni]
                    * rd_vvnn[pol_v[0], pol_v[2], nni, mmi]
                    + r_vnn[pol_v[2], mmi, nni]
                    * rd_vvnn[pol_v[0], pol_v[1], nni, mmi]) \
                    * (eta / (np.pi * ((w_l - Emn) ** 2 + eta ** 2))
                        - eta / (np.pi * ((w_l + Emn) ** 2 + eta ** 2)))

                sum2_l += fnm * tmp

    # Return the output
    return sum2_l
