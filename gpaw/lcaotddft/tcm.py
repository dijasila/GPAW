# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def generate_gridspec(**kwargs):
    from matplotlib.gridspec import GridSpec
    width = 0.84
    bottom = 0.12
    left = 0.12
    return GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                    bottom=bottom, top=bottom + width,
                    left=left, right=left + width,
                    **kwargs)


def plot_DOS(ax, energy_e, dos_e, base_e, dos_min, dos_max,
             flip=False, fill=None, line=None):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if flip:
        set_label = ax.set_xlabel
        fill_between = ax.fill_betweenx
        set_energy_lim = ax.set_ylim
        set_dos_lim = ax.set_xlim

        def plot(x, y, *args, **kwargs):
            return ax.plot(y, x, *args, **kwargs)
    else:
        set_label = ax.set_ylabel
        fill_between = ax.fill_between
        set_energy_lim = ax.set_xlim
        set_dos_lim = ax.set_ylim

        def plot(x, y, *args, **kwargs):
            return ax.plot(x, y, *args, **kwargs)
    if fill:
        fill_between(energy_e, base_e, dos_e + base_e, **fill)
    if line:
        plot(energy_e, dos_e, **line)
    set_label('DOS', labelpad=0)
    set_energy_lim(np.take(energy_e, (0, -1)))
    set_dos_lim(dos_min, dos_max)


class TCMPlotter(object):

    def __init__(self, ksd, energy_o, energy_u, sigma,
                 zero_fermilevel=True):
        self.ksd = ksd
        self.eig_n, self.fermilevel = ksd.get_eig_n(zero_fermilevel)
        self.energy_o = energy_o
        self.energy_u = energy_u
        self.base_o = np.zeros_like(energy_o)
        self.base_u = np.zeros_like(energy_u)
        self.sigma = sigma

    def plot_TCM(self, weight_p, vmax='80%'):
        gs = generate_gridspec(hspace=0.05, wspace=0.05)
        self.ax_tcm = plt.subplot(gs[2])

        # Calculate TCM
        eig_n = self.eig_n
        energy_o = self.energy_o
        energy_u = self.energy_u
        sigma = self.sigma
        fermilevel = self.fermilevel
        tcm_ou = self.ksd.get_TCM(weight_p, eig_n, energy_o, energy_u, sigma)

        # Plot TCM
        ax = self.ax_tcm
        plt.sca(ax)
        if isinstance(vmax, str):
            assert vmax[-1] == '%'
            tcmmax = np.max(np.absolute(tcm_ou))
            vmax = tcmmax * float(vmax[:-1]) / 100.0
        vmin = -vmax
        if tcm_ou.dtype == complex:
            linecolor = 'w'
            from matplotlib.colors import hsv_to_rgb

            def transform_to_hsv(z, rmin, rmax, hue_start=90):
                amp = np.absolute(z)
                amp = np.where(amp < rmin, rmin, amp)
                amp = np.where(amp > rmax, rmax, amp)
                ph = np.angle(z, deg=1) + hue_start
                h = (ph % 360) / 360
                s = 0.85 * np.ones_like(h)
                v = (amp - rmin) / (rmax - rmin)
                return hsv_to_rgb(np.dstack((h, s, v)))

            img = transform_to_hsv(tcm_ou.T, 0, vmax)
            plt.imshow(img, origin='lower',
                       extent=[energy_o[0], energy_o[-1],
                               energy_u[0], energy_u[-1]]
                       )
        else:
            linecolor = 'k'
            cmap = 'seismic'
            plt.pcolormesh(energy_o, energy_u, tcm_ou.T,
                           cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
        plt.axhline(fermilevel, c=linecolor)
        plt.axvline(fermilevel, c=linecolor)

        ax.tick_params(axis='both', which='major', pad=2)
        plt.xlabel(r'Occ. energy $\varepsilon_{o}$ (eV)', labelpad=0)
        plt.ylabel(r'Unocc. energy $\varepsilon_{u}$ (eV)', labelpad=0)
        plt.xlim(np.take(energy_o, (0, -1)))
        plt.ylim(np.take(energy_u, (0, -1)))
        return ax

    def get_axis(self, name):
        if name == 'dos':
            if hasattr(self, 'ax_occ_dos'):
                return self.ax_occ_dos, self.ax_unocc_dos
            else:
                gs = generate_gridspec(hspace=0.05, wspace=0.05)
                self.ax_occ_dos = plt.subplot(gs[0])
                self.ax_unocc_dos = plt.subplot(gs[3])
                return self.ax_occ_dos, self.ax_unocc_dos

    def plot_DOS(self, weight_n=1.0, stack=False,
                 fill={'color': '0.8'}, line={'color': 'k'}):
        ax_occ_dos, ax_unocc_dos = self.get_axis('dos')

        # Calculate DOS
        dos_o, dos_u = self.ksd.get_weighted_DOS(weight_n, self.eig_n,
                                                 self.energy_o, self.energy_u,
                                                 self.sigma)

        # Plot DOSes
        if stack:
            base_o = self.base_o
            base_u = self.base_u
        else:
            base_o = np.zeros_like(self.energy_o)
            base_u = np.zeros_like(self.energy_u)
        dos_min = 0.0
        dos_max = 1.01 * max(np.max(dos_o), np.max(dos_u))
        plot_DOS(ax_occ_dos, self.energy_o, dos_o, base_o, dos_min, dos_max,
                 flip=False, fill=fill, line=line)
        plot_DOS(ax_unocc_dos, self.energy_u, dos_u, base_u, dos_min, dos_max,
                 flip=True, fill=fill, line=line)
        if stack:
            self.base_o += dos_o
            self.base_u += dos_u
        return ax_occ_dos, ax_unocc_dos

    def plot_spectrum(self):
        gs = generate_gridspec(hspace=0.8, wspace=0.8)
        self.ax_spec = plt.subplot(gs[1])
