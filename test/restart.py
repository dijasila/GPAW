import os
from gpaw import Calculator
from ase import *
from gpaw.utilities import equal


netcdf = True
try:
    import Scientific.IO.NetCDF
except ImportError:
    netcdf = False

if 1:
    h = Atoms([Atom('H')], cell=(4, 4, 4), pbc=1)
    calc = Calculator(nbands=1, gpts=(16, 16, 16),)# out=None)
    h.set_calculator(calc)
    e = h.get_potential_energy()
    calc.write('tmp.gpw')
    if netcdf:
        calc.write('tmp.nc', 'all')

h = Calculator('tmp.gpw', txt=None)
equal(e, h.get_potential_energy(), 3e-5)

if netcdf:
    h = Calculator('tmp.nc', txt=None)
    equal(e, h.get_potential_energy(), 3e-5)

os.remove('tmp.gpw')
if netcdf:
    os.remove('tmp.nc')
