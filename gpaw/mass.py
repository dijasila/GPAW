import pickle
from math import pi
from pathlib import Path

import numpy as np
from ase.dft.bandgap import bandgap
from ase.units import Bohr, Ha
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates


def extract_stuff_from_gpw_file(path):
    out = path.with_suffix('.pckl')
    if out.is_file():
        print(f'Reading from {out}')
        return pickle.loads(out.read_bytes())

    print(f'Extracting eigenvalues and wave function projections from {path}')
    calc = GPAW(path)
    assert (calc.atoms.pbc == [False, False, True]).all()
    states = soc_eigenstates(calc)

    kpoints = np.array([calc.wfs.kd.bzk_kc[wf.bz_index]
                        for wf in states])
    eigenvalues = np.array([wf.eig_m
                            for wf in states])
    fingerprints = np.array([wf.projections.matrix.array
                             for wf in states])

    for direct in [False, True]:
        bandgap(eigenvalues=eigenvalues,
                efermi=states.fermi_level,
                direct=direct, kpts=kpoints)
        print()

    results = (kpoints[:, 2],
               calc.atoms.cell[2, 2],
               states.fermi_level,
               eigenvalues,
               fingerprints)

    out.write_bytes(pickle.dumps(results))

    return results


def connect(eigenvalues, fingerprints, eps=0.001):
    K, N = eigenvalues.shape

    for k in range(K - 1):
        overlap = abs(fingerprints[k] @ fingerprints[k + 1].conj().T)
        indices = (-overlap).argsort(axis=1)
        bands = []
        for ii in indices:
            for n in ii:
                if n not in bands:
                    bands.append(n)
                    break
            else:  # no break
                1 / 0
        eigenvalues[k + 1] = eigenvalues[k + 1, bands]
        fingerprints[k + 1] = fingerprints[k + 1, bands]

    for k in range(1, K - 1):
        eigs = eigenvalues[k]
        for bands in clusters(eigs, eps):
            overlap = abs(fingerprints[k - 1, bands] @
                          fingerprints[k + 1, bands].conj().T)
            indices = overlap.argmax(axis=1)
            # print(k, bands, overlap, indices)
            assert indices.sum() == len(bands) * (len(bands) - 1) // 2
            bands2 = [bands[i] for i in indices]
            eigenvalues[k + 1:, bands] = eigenvalues[k + 1:, bands2]
            fingerprints[k + 1:, bands] = fingerprints[k + 1:, bands2]

    return eigenvalues


def clusters(eigs, eps=1e-4):
    degenerate = abs(eigs - eigs[:, np.newaxis]) < eps
    taken = set()
    for row in degenerate:
        bands = [n for n, deg in enumerate(row) if deg and n not in taken]
        if len(bands) > 1:
            yield bands
        taken.update(bands)


def fit(kpoints,
        length,
        fermi_level,
        eigenvalues,
        fingerprints,
        kind='cbm',
        N=4,
        plot=True):

    K = len(kpoints)

    nocc = (eigenvalues[0] < fermi_level).sum()

    if kind == 'cbm':
        bands = slice(nocc, nocc + N)
        eigs = eigenvalues[:, bands] - fermi_level
    else:
        bands = slice(nocc - 1, nocc - 1 - N, -1)
        eigs = fermi_level - eigenvalues[:, bands]

    fps = fingerprints[:, bands]

    eigs2 = np.empty_like(eigs)
    fps2 = np.empty_like(fps)
    imin = eigs[:, 0].argmin()
    i0 = K // 2
    for i in range(K):
        eigs2[(i0 + i) % K] = eigs[(imin + i) % K]
        fps2[(i0 + i) % K] = fps[(imin + i) % K]
    kpoints2 = kpoints[imin] + np.linspace(0, 1, K, endpoint=False) - i0 / K
    x = 2 * pi / length * kpoints2

    eigs = connect(eigs2, fps2)

    extrema = {}
    indices = eigs[i0].argsort()
    print('k [Ang^-1]  e [eV]   m [m_e]')
    for n in indices:
        band = eigs[:, n]
        i = band.argmin()
        if 2 <= i <= K - 3:
            poly = np.polyfit(x[i - 2:i + 3], band[i - 2:i + 3], 2)
            xfit = np.linspace(x[i - 3], x[i + 3], 61)
            yfit = np.polyval(poly, xfit)
            mass = 0.5 / Bohr**2 / Ha / poly[0]
            k = -0.5 * poly[1] / poly[0]
            energy = np.polyval(poly, k)
            if kind == 'vbm':
                energy *= -1
                yfit *= -1
                band *= -1
            print(f'{k:10.3f} {energy:7.3f} {mass:8.3f}')
            extrema[n] = (xfit, yfit, mass, k, energy)

    if plot:
        import matplotlib.pyplot as plt
        color = 0
        for n in indices:
            plt.plot(x, eigs[:, n], 'o', color=f'C{color}')
            if n in extrema:
                xfit, yfit, _, _, _ = extrema[n]
                plt.plot(xfit, yfit, '-', color=f'C{color}')
            color += 1
        plt.xlabel('k [Ang$^{-1}$]')
        plt.ylabel('e - e$_F$ [eV]')
        plt.show()

    return [(mass, k, energy)
            for (_, _, mass, k, energy) in extrema.values()]


def test():
    k = np.linspace(-1, 1, 7)
    b1 = (k - 0.2)**2
    b2 = 1 * (k + 0.2)**2 + 0.01 * 0
    eigs = np.array([b1, b2]).T
    indices = eigs.argsort(axis=1)
    eigs = np.take_along_axis(eigs, indices, axis=1)
    fps = np.zeros((7, 2, 2))
    fps[:, 0, 0] = 1
    fps[:, 1, 1] = 1
    fps[3] = 0.0
    fps[:, :, 0] = np.take_along_axis(fps[:, :, 0], indices, axis=1)
    fps[:, :, 1] = np.take_along_axis(fps[:, :, 1], indices, axis=1)
    eigs = fit(k, eigs, fps)
    import matplotlib.pyplot as plt
    plt.plot(k, eigs[:, 0])
    plt.plot(k, eigs[:, 1])
    plt.show()


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1])
    kpoints, length, fermi_level, eigenvalues, fingerprints = \
        extract_stuff_from_gpw_file(path)
    fit(kpoints, length, fermi_level,
        eigenvalues, fingerprints, kind=sys.argv[2])
    