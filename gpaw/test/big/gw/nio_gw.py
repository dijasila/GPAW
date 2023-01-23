"""AFM NiO: Make sure spin-polarized GW works."""
from gpaw.response.g0w0 import G0W0

gw = G0W0('nio.gpw',
          'gw6ok',
          nbands=100,
          ecut=100,
          truncation='wigner-seitz',
          kpts=[(0, 0, 0)],
          relbands=(-1, 1),
          nblocks=4)
result = gw.calculate()

# Make sure gaps in both spin-channels are the same:
pbe_sn = result['eps'][:, 0]
pbe_gap_s = pbe_sn[:, 1] - pbe_sn[:, 0]
assert abs(pbe_gap_s - 2.319).max() < 0.01
qp_sn = result['qp'][:, 0]
qp_gap_s = qp_sn[:, 1] - qp_sn[:, 0]
assert abs(qp_gap_s - 3.743).max() < 0.01
