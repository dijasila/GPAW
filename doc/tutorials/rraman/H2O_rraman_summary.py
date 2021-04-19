from ase import io
from ase.vibrations.placzek import Placzek
from gpaw.lrtddft import LrTDDFT

atoms = io.read('relaxed.traj')
pz = Placzek(atoms, LrTDDFT,
             name='ir',  # use ir-calculation for frequencies
             exname='rraman_erange17'  # use LrTDDFT for intensities
)
omega = 0  # excitation frequency
gamma = 0.2  # width
pz.summary(omega, gamma)
