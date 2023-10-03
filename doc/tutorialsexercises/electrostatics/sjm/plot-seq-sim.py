# web-page: simultaneous.png

from matplotlib import pyplot, gridspec


def makefig():
    fig = pyplot.figure(figsize=(6.5, 6.5))
    grid = gridspec.GridSpec(2, 1, fig, right=0.95, top=0.95)
    smallgrid = gridspec.GridSpecFromSubplotSpec(2, 1, grid[0], hspace=0.)
    ax0 = fig.add_subplot(smallgrid[0])
    ax1 = fig.add_subplot(smallgrid[1], sharex=ax0)
    smallgrid = gridspec.GridSpecFromSubplotSpec(2, 1, grid[1], hspace=0.)
    ax2 = fig.add_subplot(smallgrid[0], sharex=ax0)
    ax3 = fig.add_subplot(smallgrid[1], sharex=ax0)
    return fig, ax0, ax1, ax2, ax3


def get_data(mode):
    with open(f'Au111-H-{mode}.txt', 'r') as f:
        lines = f.read().splitlines()
    potentials = []
    equilibrateds = []
    fmaxs = []
    for line in lines:
        if 'Potential found to be' in line:
            potentials += [float(line.split('to be')[-1].split('V (with')[0])]
        if 'Potential is within tolerance. Equilibrated.' in line:
            diff = [False] * (len(potentials) - len(equilibrateds))
            diff[-1] = True
            equilibrateds.extend(diff)
    with open(f'qn-Au111-H-{mode}.log', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        if 'BFGS:' in line:
            fmaxs += [float(line.split()[-1])]
    return potentials, equilibrateds, fmaxs


def plot_data(potentials, equilibrateds, fmaxs, axes):
    fmax_indices = []
    axes[0].plot(potentials, 'o-', color='k', markerfacecolor='w')
    for index, (potential, equilibrated) in enumerate(zip(potentials,
                                                          equilibrateds)):
        if equilibrated:
            axes[0].plot(index, potential, 'og', markeredgecolor='k')
            fmax_indices += [index]
    axes[1].plot(fmax_indices, fmaxs, 'ko-')


def prune_duplicate(potentials, equilibrateds):
    """Due to logging, there's a duplicate potential reading when
    we switched from simultaneous to sequential. Remove it, and also
    return the image number of the switch."""
    for index, potential in enumerate(potentials):
        if index > 0 and potential == potentials[index - 1]:
            duplicate = index
    del potentials[duplicate]
    del equilibrateds[duplicate]
    return duplicate


def shade_plot(ax, xstart):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(x=[xstart, xlim[-1]], y1=[ylim[0]] * 2, y2=[ylim[1]] * 2,
                    color='0.8')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


fig, ax0, ax1, ax2, ax3 = makefig()
potentials, equilibrateds, fmaxs = get_data('seq')
plot_data(potentials, equilibrateds, fmaxs, [ax0, ax1])
potentials, equilibrateds, fmaxs = get_data('sim')
duplicate = prune_duplicate(potentials, equilibrateds)
plot_data(potentials, equilibrateds, fmaxs, [ax2, ax3])

ax3.set_xlabel('DFT calculation')
ax0.text(0.20, 0.75, 'sequential', transform=ax0.transAxes)
ax2.text(0.20, 0.75, 'simultaneous', transform=ax2.transAxes)
ax2.text(0.75, 0.75, '(sequential)', transform=ax2.transAxes)
shade_plot(ax2, duplicate - 0.5)
shade_plot(ax3, duplicate - 0.5)
for ax in [ax0, ax2]:
    ax.set_ylabel(r'$\phi$, V')
for ax in [ax1, ax3]:
    ax.set_ylabel(r'max force, eV/$\mathrm{\AA}$')

fig.savefig('simultaneous.png')
