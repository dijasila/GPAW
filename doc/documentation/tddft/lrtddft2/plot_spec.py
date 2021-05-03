# web-page: abs_spec.png, rot_spec.png
import numpy as np
import matplotlib.pyplot as plt

for name, specj in zip(['abs', 'rot'], [1, 2]):
    plt.figure(figsize=(6, 6 / 1.62))
    ax = plt.subplot(1, 1, 1)
    for ecut, ls in zip([7, 8], ['k--', 'k']):
        data_ej = np.loadtxt('spectrum_with_%05.2feV.dat' % ecut)
        ax.plot(data_ej[:, 0], data_ej[:, specj], ls, label=f'{ecut:.2f} eV')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Energy (eV)')
    plt.legend(title='max_energy_diff')
    if name == 'abs':
        plt.ylabel('Photoabsorption (eV$^{-1}$)')
    elif name == 'rot':
        plt.ylabel('Rotatory strength (10$^{-40}$ cgs eV$^{-1}$)')
    plt.xlim(5, 10)
    plt.tight_layout()
    plt.savefig(f'{name}_spec.png')
