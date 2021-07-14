import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor


def test_smooth():
    rgd = EquidistantRadialGridDescriptor(0.05, 200)
    r = rgd.r_g
    a = 3.0
    g = np.exp(-a * r) * r
    g[40:] = 0.0
    g2, c2 = rgd.pseudize(g, 20, 1)
    g3, c3 = rgd.pseudize_smooth(g, 20, 1)
    print(c2, c3)
    import matplotlib.pyplot as plt
    plt.plot(r, g)
    plt.plot(r, g2)
    plt.plot(r, g3)
    plt.xlim(0, 2)
    plt.show()
