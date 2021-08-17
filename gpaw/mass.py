import pickle
from math import pi
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
from ase.dft.bandgap import bandgap
from ase.units import Bohr, Ha

from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates
from gpaw.typing import Array1D, Array2D, Array3D


def extract_stuff_from_gpw_file(path: Path) -> Tuple[Array1D,
                                                     float, float,
                                                     Array2D,
                                                     Array3D,
                                                     Array3D]:
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
    spinprojections = states.spin_projections()

    for direct in [False, True]:
        bandgap(eigenvalues=eigenvalues,
                efermi=states.fermi_level,
                direct=direct, kpts=kpoints)
        print()

    results = (kpoints[:, 2],
               calc.atoms.cell[2, 2],
               states.fermi_level,
               eigenvalues,
               fingerprints,
               spinprojections)

    out.write_bytes(pickle.dumps(results))

    return results


def connect(eigenvalues: Array2D,
            fingerprints: Array3D,
            spinprojections: Array3D = None,
            eps: float = 0.001) -> Array2D:
    """
    >>> eigs = np.array([[0, 1], [0, 1]])
    >>> fps = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    >>> connect(eigs, fps)
    array([[0, 1],
           [1, 0]])
    """
    K, N = eigenvalues.shape

    eigenvalues = eigenvalues.copy()
    fingerprints = fingerprints.copy()

    if spinprojections is not None:
        spinprojections = spinprojections.copy()

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
        if spinprojections is not None:
            spinprojections[k + 1] = spinprojections[k + 1, bands]

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
            if spinprojections is not None:
                spinprojections[k + 1:, bands] = spinprojections[k + 1:,
                                                                 bands2]

    return eigenvalues, spinprojections


def clusters(eigs: Array1D,
             eps: float = 1e-4) -> Generator[List[int], None, None]:
    """
    >>> list(clusters(np.zeros(4)))
    [[0, 1, 2, 3]]
    >>> list(clusters(np.arange(4)))
    []
    >>> list(clusters(np.array([0, 0, 1, 1, 1, 2])))
    [[0, 1], [2, 3, 4]]
    """
    degenerate = abs(eigs - eigs[:, np.newaxis]) < eps
    taken = set()
    for row in degenerate:
        bands = [n for n, deg in enumerate(row) if deg and n not in taken]
        if len(bands) > 1:
            yield bands
        taken.update(bands)


def fit(kpoints: Array1D,
        fermi_level: float,
        eigenvalues: Array2D,
        fingerprints: Array3D,
        spinprojections: Array3D = None,
        kind: str = 'cbm',
        N: int = 4,
        plot: bool = True) -> List[Tuple[float, float, float, Array1D]]:
    """...

    >>> k = np.linspace(-1, 1, 7)
    >>> eigs = 0.5 * k**2 * Ha * Bohr**2
    >>> minima = fit(kpoints=k,
    ...              fermi_level=-1.0,
    ...              eigenvalues=eigs[:, np.newaxis],
    ...              fingerprints=np.zeros((7, 1, 1)),
    ...              kind='cbm',
    ...              N=1,
    ...              plot=False)
    k [Ang^-1]  e-e_F [eV]    m [m_e]
        -0.000       1.000      1.000
    >>> k0, e0, m0 = minima[0]
    """

    K = len(kpoints)

    nocc = (eigenvalues[0] < fermi_level).sum()

    if kind == 'cbm':
        bands = slice(nocc, nocc + N)
        eigs = eigenvalues[:, bands] - fermi_level
    else:
        bands = slice(nocc - 1, nocc - 1 - N, -1)
        eigs = fermi_level - eigenvalues[:, bands]

    fps = fingerprints[:, bands]
    sps = spinprojections[:, bands]

    eigs2 = np.empty_like(eigs)
    fps2 = np.empty_like(fps)
    sps2 = np.empty_like(sps)
    imin = eigs[:, 0].argmin()
    i0 = K // 2
    for i in range(K):
        eigs2[(i0 + i) % K] = eigs[(imin + i) % K]
        fps2[(i0 + i) % K] = fps[(imin + i) % K]
        sps2[(i0 + i) % K] = sps[(imin + i) % K]
    x = kpoints[imin] + kpoints - kpoints[i0]

    eigs, sps = connect(eigs2, fps2, sps2)

    extrema = {}
    indices = eigs[i0].argsort()
    print('k [Ang^-1]  e-e_F [eV]    m [m_e]            spin [x,y,z]')
    for n in indices:
        band = eigs[:, n]
        i = band.argmin()
        if 2 <= i <= K - 3:
            poly = np.polyfit(x[i - 2:i + 3], band[i - 2:i + 3], 2)
            dx = 1.5 * (x[i + 2] - x[i])
            xfit = np.linspace(x[i] - dx, x[i] + dx, 61)
            yfit = np.polyval(poly, xfit)
            mass = 0.5 * Bohr**2 * Ha / poly[0]
            assert mass > 0
            k = -0.5 * poly[1] / poly[0]
            energy = np.polyval(poly, k)
            if kind == 'vbm':
                energy *= -1
                yfit *= -1
            spin = sps[i, n]
            print(f'{k:10.3f} {energy:11.3f} {mass:10.3f}',
                  '  (' + ', '.join(f'{s:+.2f}' for s in spin) + ')')
            extrema[n] = (xfit, yfit, k, energy, mass, spin)

    if kind == 'vbm':
        eigs *= -1

    if plot:
        import matplotlib.pyplot as plt
        color = 0
        for n in indices:
            plt.plot(x, eigs[:, n], 'o', color=f'C{color}')
            if n in extrema:
                xfit, yfit, *_ = extrema[n]
                plt.plot(xfit, yfit, '-', color=f'C{color}')
            color += 1
        plt.xlabel('k [Ang$^{-1}$]')
        plt.ylabel('e - e$_F$ [eV]')
        plt.show()

    return [(k, energy, mass, spin)
            for (_, _, k, energy, mass, spin) in extrema.values()]


def a_test():
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
    (kpoints, length, fermi_level,
     eigenvalues, fingerprints,
     spinprojections) = extract_stuff_from_gpw_file(path)
    extrema = fit(kpoints * 2 * pi / length,
                  fermi_level,
                  eigenvalues,
                  fingerprints,
                  spinprojections,
                  kind=sys.argv[2])
