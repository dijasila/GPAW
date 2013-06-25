import numpy as np
from ase.structure import bulk
from gpaw import GPAW, PW, FermiDirac, MethfesselPaxton

a0 = 2.84
fe = bulk('Fe', 'bcc', a=a0)
fe[0].magmom = 2.3
cell0 = fe.cell

for ecut in [300, 400, 500, 600, 700, 800]:
    fe.calc = GPAW(mode=PW(ecut),
                   xc='PBE',
                   kpts=(8, 8, 8),
                   parallel={'band': 1},
                   basis='dzp',
                   txt='Fe-%d.txt' % ecut)
    for eps in np.linspace(-0.02, 0.02, 5):
        fe.cell = (1 + eps) * cell0
        fe.get_potential_energy()

fe.calc.set(mode=PW(800))
for k in range(4, 13):
    fe.calc.set(kpts=(k,k,k))
    for width in [0.05, 0.1, 0.15, 0.2]:
        for name, occ in [('FD', FermiDirac(width)),
                          ('MP', MethfesselPaxton(width))]:
            fe.calc.set(occupations=occ,
                        txt='Fe-%02d-%s-%.2f.txt' % (k, name, width))
            for eps in np.linspace(-0.02, 0.02, 5):
                fe.cell = (1 + eps) * cell0
                fe.get_potential_energy()
