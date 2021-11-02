from gpaw.response.g0w0 import G0W0

gw = G0W0('nio.gpw',
          nbands=100,
          ecut=100,
          wstc=True,
          kpts=[(0, 0, 0)],
          relbands=(-1, 1))
result = gw.calculate()
