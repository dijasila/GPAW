# web-page: Polarised_raman_488nm.png
import numpy as np
from gpaw.elph import RamanData

rd = RamanData()
entries = [["xx", "yy"], ["xy", "yx"], ["xz", "zx"], ["zz", ]]

spectra = []
labels = []
for entry in entries:
    energy, spectrum = rd.calculate_raman_spectrum(entry)
    spectra.append(spectrum)
    if len(entry) == 2:
        label = entry[0] + "+" + entry[1]
    else:
        label = entry[0]
    labels.append(label)

rd.plot_raman("Polarised_raman_488nm.png", energy, spectra, labels,
              relative=False)

# for testing
np.save("raman_spectrum.npy", np.vstack((energy, spectra)))
