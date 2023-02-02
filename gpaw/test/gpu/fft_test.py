def wip_test_fft():
    L = 5
    n = 20
    """
    grid = UniformGrid(cell=[L, L, L], size=(n, n, n), comm=world)
    f = grid.zeros()


    pw = PlaneWaves(cell=grid.cell, ecut=700, comm=world)
    f2 = f.fft(pw=pw)

    assert abs(f2.integrate()) < 1e-14
    assert f2.moment() == pytest.approx([0, moment, 0], abs=1e-5)
    """
