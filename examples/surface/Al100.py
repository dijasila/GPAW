from gpaw import Calculator
from build import fcc100

a = 4.05
def energy(n):
    fcc = fcc100('Al', a, n, 20.0)
    calc = Calculator(nbands=n * 5,
                      kpts=(6, 6, 1),
                      h = 0.25)
    fcc.set_calculator(calc)
    return fcc.get_potential_energy()

f = file('e6x6.dat', 'w')
for n in range(3, 7):
    e = energy(n)
    print >> f, n, e
