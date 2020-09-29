
# Import required modules: General
from gpaw.mpi import world, broadcast
import sys
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from pathlib import Path


# Prints only from master


def parprint(*args, **kwargs):
    """MPI-safe print - prints only from master. """
    if world.rank == 0:
        print(*args, **kwargs)
        sys.stdout.flush()

# Print iterations progress


def print_progressbar(iteration, total, prefix='Progress:', suffix='Complete',
                      decimals=1, length=50, fill='#', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals complete
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if world.rank == 0:
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(
            '\r%s |%s| %s%% %s' %
            (prefix, bar, percent, suffix), end=printEnd)
        sys.stdout.flush()

        # Print New Line on Complete
        if iteration == total:
            print()
            sys.stdout.flush()

# Plot an spectrum


def plot_spectrum(figname='response.png', dtype='abs',
                  wlim=None, ylim=None, mult=1.0,
                  resname=None, pind=None,
                  leg=None, legloc='best',
                  title=None, ylabel=None):
    """
    Plot a given response spectrum

    Input:
        figname     Name of the generated figure
        dtype       String for choosing the plot type (abs, re, or im)
        wlim        The plotting frequency range as a 2 element tuple
        ylim        The plotting y range as a 2 element tuple
        mult        Multiply data by a factor (default 1.0)
        resname     Suffix used for files containing the spectra (name or list)
        pind        Selected index of the data (in the numpy array)
        leg         List of legend (optional)
        legloc      Legend location
        title       Figure title
        ylabel      Label for the y direction

    Output:
        figure      Image containing the spectrum.
    """

    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['font.size'] = 10

    # Plot the figure in one core
    if world.rank == 0:
        # Make the figure frame
        figW = 6.0
        figH = 4.0
        dpiVal = 300
        plt.figure(figsize=(figW, figH), dpi=dpiVal)

        if resname is None:
            legend = False
            res_name = ['df.npy']
        elif type(resname) == list:
            legend = True
            res_name = ['{}.npy'.format(name) for name in resname]
        else:
            legend = False
            res_name = ['{}.npy'.format(resname)]

        # Plot the data
        legls = []
        # Loop over response files
        for ii, name in enumerate(res_name):
            # Load the data
            alldata = np.load(name)
            if pind is None:
                pindi = range(1, len(alldata))
            else:
                pindi = pind
            w_l = alldata[0]
            # Loop over the data in response (if there are several columns)
            for ind, data in enumerate(alldata[pindi]):
                # Depending of what is required
                if dtype == 'abs':
                    plt.plot(np.real(w_l), np.abs(mult * data))
                    legls.append('C{}: {}'.format(ind + 1, name))
                    ylab = 'abs'
                elif dtype == 're':
                    plt.plot(np.real(w_l), np.real(mult * data))
                    legls.append('C{}: {}'.format(ind + 1, name))
                    ylab = r'$\Re$'
                elif dtype == 'im':
                    plt.plot(np.real(w_l), np.imag(mult * data))
                    legls.append('C{}: {}'.format(ind + 1, name))
                    ylab = r'$\Im$'
                elif dtype == 'comp':
                    plt.plot(np.real(w_l), np.real(mult * data), '-')
                    plt.plot(np.real(w_l), np.imag(mult * data), '--')
                    legls.append('C{}: Re, {}'.format(ind + 1, name))
                    legls.append('C{}: Im, {}'.format(ind + 1, name))
                    ylab = ''
                else:
                    raise 'Error in dtype'

        # Put the limits and legends, and finally save it
        if wlim is not None:
            plt.xlim(wlim)
        if ylim is not None:
            plt.ylim(ylim)
        if legend is True:
            if leg is None:
                plt.legend(legls, loc=legloc)
            else:
                plt.legend(leg, loc=legloc)
        if title is not None:
            # plt.title('\n'.join(wrap(title, 60)))
            plt.title(title)
        plt.xlabel(r'$\hbar\omega$ (eV)')
        if ylabel is None:
            plt.ylabel(r'{}[response (SI unit)]'.format(ylab))
        else:
            plt.ylabel(ylab + '[' + ylabel + ']')

        # Save the figure
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.clf()
        plt.close()

# Plot a polarized map


def plot_polar(psi, rdata, fig=None, figname='polarized.png',
               leg=None, legloc='best', title=None):
    """
    Plot a polar graph

    Input:
        psi         Angle vector
        rdata       Abs vector or a lsit of Abs vector
        figname     Name of the generated figure
        leg         List of legend (optional)
        legloc      Legend location
        title       Figure title

    Output:
        figure      Image containing the polar graph.
    """

    # Plot the figure in one core
    if world.rank == 0:
        # Make the figure frame
        if fig is None:
            figW = 4.0
            figH = 4.0
            fig = plt.figure(figsize=(figW, figH), dpi=300)
        ax = plt.subplot(111, projection='polar')
        if type(rdata) == list:
            # Loop over the data in response (if there is list)
            for ind, cdata in enumerate(rdata):
                ax.plot(psi, cdata)
        else:
            ax.plot(psi, rdata)
        ax.grid(True)
        if title is not None:
            ax.set_title(title)
        if leg is not None:
            if type(rdata) == list:
                assert len(leg) == len(
                    rdata), 'The length og legend is not correct'
            plt.legend(leg, loc=legloc)

        # Save the figure
        plt.tight_layout()
        if figname is None:
            return plt.gcf()
        else:
            plt.savefig(figname, dpi=300)
            plt.clf()
            plt.close()

# Plot an spectrum


def plot_kmesh(kd, icell_cv, figname='kmesh.png'):
    """
    Plot the k mesh used for integration

    Input:
        figname: Name of the generated figure
        dtype: string for choosing to plot abs, re, or im
        wlim: The plotting frequency range as a 2 element tuple
        ylim: The plotting y range as a 2 element tuple
        resname: Suffix used for the file containing the response spectrum
        pind: Selected index of the data

    Output:
        figname: image containing the k mesh.

    """

    if kd is None:
        return

    # Plot the figure in one core
    if world.rank == 0:

        # Make the figure frame
        figW = 6.0
        figH = 4.0
        dpiVal = 300
        plt.figure(figsize=(figW, figH), dpi=dpiVal)

        bzk_kc = kd.bzk_kc
        ibzk_kc = kd.ibzk_kc
        # bz2ibz_k = kd.bz2ibz_k
        # ibz2bz_k = kd.ibz2bz_k
        nk = len(bzk_kc[:, 0])
        w_k = kd.weight_k
        b1 = 2 * pi * icell_cv[0]
        b2 = 2 * pi * icell_cv[1]
        bzk_kc2 = np.dot(ibzk_kc, 2 * pi * icell_cv)
        # bzk_kc2 = bzk_kc
        plt.arrow(0, 0, b1[0], b1[1], color='r')
        plt.arrow(0, 0, b2[0], b2[1], color='r')
        plt.scatter(bzk_kc2[:, 0], bzk_kc2[:, 1], s=2 * w_k * nk)

        plt.axis('equal')
        # plt.xlim((-0.5, 0.5))
        # plt.ylim((-0.5, 0.5))
        plt.savefig(figname, dpi=300)
        plt.clf()


# Check if the file exist or not


def is_file_exist(filename):

    if world.rank == 0:
        if (not Path(filename).is_file()):
            calc_required = True
        else:
            calc_required = False
    else:
        calc_required = None
    # calc_required = par.comm.bcast(calc_required, root=0)
    # calc_required = world.broadcast(calc_required, 0)
    calc_required = broadcast(calc_required, 0)
    return calc_required


# def plot_kfunction(kd, fk, figname='output.png',
#                    tsymm='even', dtype='re', clim=None):

#     # Useful variables
#     if world.rank == 0:
#         psigns = -2 * kd.time_reversal_k + 1
#         N_c = kd.N_c
#         kx = np.reshape(kd.bzk_kc[:, 0], (N_c[0], N_c[1]))
#         ky = np.reshape(kd.bzk_kc[:, 1], (N_c[0], N_c[1]))

#         figW = 6.0
#         figH = 4.0
#         dpiVal = 300
#         plt.figure(figsize=(figW, figH), dpi=dpiVal)

#         # Plot the color map
#         if tsymm == 'even':
#             zk = np.reshape(fk[kd.bz2ibz_k], (N_c[0], N_c[1]))
#         else:
#             zk = np.reshape(fk[kd.bz2ibz_k] * psigns, (N_c[0], N_c[1]))
#         if dtype == 're':
#             zk = np.real(zk)
#         elif dtype == 'im':
#             zk = np.imag(zk)
#         elif dtype == 'abs':
#             zk = np.abs(zk)
#         else:
#             raise 'Error in dtype'
#         plt.pcolor(kx, ky, zk)
#         if clim is not None:
#             plt.clim(clim)
#         plt.colorbar()

#         # Save the figure
#         plt.tight_layout()
#         plt.savefig(figname, dpi=dpiVal)
#         plt.clf()
#         plt.close()
