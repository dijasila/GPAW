from gpaw.tddft import TDDFT, photoabsorption_spectrum
from gpaw.inducedfield import TDDFTInducedField

# Calculate photoabsorption spectrum as usual
folding = 'Gauss'
width = 0.1
photoabsorption_spectrum('na2_td_dm.dat', 'na2_td_spectrum_x.dat',
                         folding=folding, width=width)

# Load TDDFT object
td_calc = TDDFT('na2_td.gpw')

# Load InducedField object
ind = TDDFTInducedField(filename='na2_td.ind',
                        paw=td_calc)

# Calculate induced electric field
ind.calculate_induced_field(gridrefinement=2)

# Save induced electric field
ind.write('na2_td_field.ind', mode='all')
