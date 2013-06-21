def agts(queue):
    iron = queue.add('iron.py', ncpus=8, walltime=12 * 60)
    queue.add('iron.agts.py', deps=[iron],
              creates=['Fe_conv_ecut.png', 'Fe_conv_k.png'])

if __name__ == '__main__':
    import pylab as plt
    from ase.utils.eos import EquationOfState
    from ase.io import read

    def fit(filename):
        configs = read(filename + '@:')
        volumes = [a.get_volume() for a in configs]
        energies = [a.get_potential_energy() for a in configs]
        eos = EquationOfState(volumes, energies)
        v0, e0, B = eos.fit()
        return (2 * v0)**(1 / 3.0)

    cutoffs = [300, 400, 500, 600, 700, 800]
    a = [fit('Fe-%d.txt' % ecut) for ecut in cutoffs]
    plt.figure(figsize=(6, 4))
    plt.plot(cutoffs, a, 'o-')
    plt.xlabel('Plane-wave cutoff energy [eV]')
    plt.ylabel('lattice constant [Ang]')
    plt.savefig('Fe_conv_ecut.png')

    kpoints = range(4, 13)
    plt.figure(figsize=(6, 4))
    ls = '-'
    for name in ['FD', 'MP']:
        for width in [0.05, 0.1, 0.15, 0.2]:
            a = [fit('Fe-%02d-%s-%.2f.txt' % (k, name, width))
                 for k in kpoints]
            plt.plot(kpoints, a, ls=ls, label='%s-%.2f' % (name, width))
        ls = '--'
    plt.legend(loc='best')
    plt.xlabel('number of k-points')
    plt.ylabel('lattice constant [Ang]')
    plt.savefig('Fe_conv_k.png')
