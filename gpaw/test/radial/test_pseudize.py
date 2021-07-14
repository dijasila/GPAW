import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor


def test_smooth():
    rgd = EquidistantRadialGridDescriptor(0.05, 200)
    r = rgd.r_g
    a = 2.0
    f0 = np.exp(-a * r) * r
    f0[30:] = 0.0
    ecut = 20
    xcut = (2 * ecut)**0.5
    l = 1
    f1, _ = rgd.pseudize_normalized(f0, 20, l, 4)
    f2, _ = rgd.pseudize(f0, 20, l, 4)
    f3, _ = rgd.pseudize_smooth(f0, 20, l, 4, ecut)
    weights = []
    import matplotlib.pyplot as plt
    i = 1
    for f in [f0, f1, f2, f3]:
        x, ft = rgd.fft(f * r, l=0)
        weights.append(((x * ft)[x > xcut]**2).sum())
        plt.plot(r, f, label=str(i))
        i += 1
    print(weights)
    plt.xlim(0, 2)
    plt.legend()
    plt.show()
    assert (np.diff(weights) < 0).all()
