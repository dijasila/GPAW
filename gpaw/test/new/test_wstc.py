import numpy as np
from gpaw.core import UGDesc, PWDesc
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.mpi import world


def test_wstc():
    a = 10.0
    n = 80
    grid = UGDesc(cell=[a, a, a], size=[n, n, n], comm=world)
    pw = PWDesc(ecut=25, cell=grid.cell, comm=world)
    wstc = WignerSeitzTruncatedCoulomb(
        pw.cell_cv, np.array([1, 1, 1]))
    v_G = wstc.get_potential_new(pw, grid)
    v_R = grid.empty()
    v_G.ifft(out=v_R)
    print(v_G)
    print(v_R)
    print(v_R.data[0, 0])
    import matplotlib.pyplot as plt
    r = np.linspace(0, a, n, False)
    plt.plot(r, v_R.data[0, 0] * r / a**3)
    plt.show()


test_wstc()
