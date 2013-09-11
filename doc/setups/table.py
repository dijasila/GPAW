import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from ase.data import chemical_symbols
from gpaw.atom.generator2 import get_number_of_electrons


T = np.arange(87, dtype=float) - 1
T[2:] += 16
T[5:] += 10
T[13:] += 10
T[58:] += 18
T[72:] -= 18 + 14
X = T % 18
Y = 6 - T // 18
Y[Y > 0] += 0.2

def table(data, title):
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111)

    patches = []
    colors = []
    y0 = 7.4
    for t in ['std', 'sc']:
        for Z in range(1, 87):
            x, y = X[Z], Y[Z] + y0
            d = data.get((chemical_symbols[Z], t))
            if d is None or np.isnan(d):
                continue
            colors.append(abs(d))
            polygon = Polygon([(x, y), (x + 1, y),
            (x + 1, y + 1), (x, y + 1)], True)
            patches.append(polygon)
        y0 = 0
        
    p = PatchCollection(patches, cmap=mpl.cm.jet,
    #norm=mpl.colors.LogNorm(0.1,100),
                        norm=mpl.colors.LogNorm(),
                        alpha=0.6,lw=1)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.colorbar(p, orientation='horizontal', pad=0.02)
    e = 0.05
    for Z in range(1, 86):
        x = X[Z]
        y = Y[Z]
        s = chemical_symbols[Z]
        n = get_number_of_electrons(s, 'default')
        plt.text(x + e, 7.4 + y + 1 - e, s,
                 ha='left', va='top')
        plt.text(x + 1 - e, 7.4 + y + e, n,
                 ha='right', va='bottom')
        if (s, 'sc') not in data:
            continue
        n = get_number_of_electrons(s, 'semicore')
        plt.text(x + e, y + 1 - e, s,
                 ha='left', va='top')
        plt.text(x + 1 - e, y + e, n,
                 ha='right', va='bottom')
    plt.axis('off')
    plt.axis('equal')
    plt.axis(xmax=18,ymax=14.2)
    plt.title(title)
    plt.show()
    return fig


if __name__ == '__main__':
    fig=table(range(1,87),range(1,87), 'test')
    fig.show()
