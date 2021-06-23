# web-page: H2O_rraman_spectrum.png
import matplotlib.pyplot as plt
from ase import io
from ase.vibrations.placzek import Placzek
from gpaw.lrtddft import LrTDDFT

atoms = io.read('relaxed.traj')
pz = Placzek(atoms, LrTDDFT,
             name='ir',  # use ir-calculation for frequencies
             exname='rraman_erange17')  # use LrTDDFT for intensities

gamma = 0.2  # width

for i, omega in enumerate([2.41, 8]):  # photon energies
    plt.subplot(211 + i)
    x, y = pz.get_spectrum(omega, gamma,
                           start=1000, end=4000, type='Lorentzian')
    plt.text(0.1, 0.8, f'{omega} eV', transform=plt.gca().transAxes)
    plt.plot(x, y)
    plt.ylabel('cross section')

plt.xlabel('frequency [cm$^{-1}$]')
plt.savefig('H2O_rraman_spectrum.png')
