from ase.parallel import paropen
from gpaw.xc.rpa import RPACorrelation

ds = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 10.0]
ecut = 200

for d in ds:
    rpa = RPACorrelation(f'gs_{d}.gpw', txt=f'rpa_{ecut}_{d}.txt')
    E_rpa = rpa.calculate(ecut=[ecut],
                          frequency_scale=2.5,
                          skip_gamma=True,
                          filename=f'restart_{ecut}_{d}.txt')

    f = paropen(f'rpa_{ecut}.dat', 'a')
    print(d, E_rpa, file=f)
    f.close()
