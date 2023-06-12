from matplotlib import pyplot as plt
import numpy as np
for dataid in [1]:
    fx, fy = 2, 2
    f = 1
    fig1, axs1 = plt.subplots(fy, fx, sharex='col', constrained_layout=True)
    fig1.set_size_inches(f * 4 * fx, f * 3 * fy)
    k = 0
    k2 = 0
    for pre, label, fname in [(1, 'Full frequency', f'k_{k}_k2_{k2}_Wmodel_ppaFalse_mpaFalse.txt'),
                              (1, 'Multipole', f'k_{k}_k2_{k2}_Wmodel_ppaFalse_mpaTrue.txt')]:
                             #  (1, 'Plasmon pole', 'Wmodel_ppaTrue_mpaFalse.txt')]:
        for f in [0,1]:
            data = np.loadtxt(fname)
            sign = data[:, 0]
            mask = sign == f
            w = data[mask, 1]
            if dataid == 0:
                sigma = data[mask, 6] + 1j*data[mask,7]  # (2,3)
            elif dataid == 1:
                sigma = data[mask, 2] + 1j*data[mask,3]  # (0,0)
            else:
                sigma = data[mask, 4] + 1j*data[mask,5]  # (0,1)
            axs1[0,f].plot(w, sigma.real, label=f'{label} f={f}')
            axs1[1,f].plot(w, pre*sigma.imag, label=f'{label} f={f}')
    for x in range(2):
        for y in range(2):
            axs1[x,y].legend(edgecolor='k')


    plt.show()
