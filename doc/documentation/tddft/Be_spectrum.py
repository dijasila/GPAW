from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft import photoabsorption_spectrum

lr = LrTDDFT.read('lr.dat.gz')
lr.diagonalize()

# write the spectrum to the data file
photoabsorption_spectrum(lr,
                         'spectrum_w.05eV.dat',  # data file name
                         width=0.05)             # width in eV
