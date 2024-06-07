import numpy as np

from ase.io import write
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.inducedfield.inducedfield_base import BaseInducedField

# Load the objects
calc = GPAW('unocc.gpw', txt=None)
calc.initialize_positions()  # Initialize in order to calculate density
dmat = DensityMatrix(calc)
gd = dmat.density.finegd
fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')
td_calc = LCAOTDDFT('td.gpw', txt=None)


def do(w):
    # Select the frequency and the density matrix
    rho_MM = fdm.FReDrho_wuMM[w][0]
    freq = fdm.freq_w[w]
    frequency = freq.freq * au_to_eV
    print(f'Frequency: {frequency:.2f} eV')
    print(f'Folding: {freq.folding}')

    # Induced density
    rho_g = (dmat.get_density([rho_MM.real])
             + 1.0j * dmat.get_density([rho_MM.imag]))

    # Save as a cube file
    big_g = gd.collect(rho_g)
    if world.rank == 0:
        write(f'ind_{freq.freq * au_to_eV:.2f}.cube', calc.atoms,
              data=big_g.imag)

    # Calculate dipole moment for reference
    dm_v = gd.calculate_dipole_moment(rho_g.imag, center=True)
    absorption = (2 * freq.freq / np.pi * dm_v[0] / au_to_eV
                  / td_calc.kick_strength[0])
    print(f'Total absorption: {absorption:.2f} eV^-1')

    # Induced field enhancement
    ind = BaseInducedField(paw=td_calc,
                           frequencies=[frequency],
                           folding=freq.folding.folding,
                           width=freq.folding.width * au_to_eV,
                           )
    ind.Fbgef_v = td_calc.kick_strength
    ind.Frho_wg = np.array([rho_g])
    ind.field_from_density = 'comp'
    ind.fieldgd = gd
    ind.has_field = True
    # Adjust extend_N_cd case by case
    # to increase the amount of vacuum and reduce boundary artifacts
    ind.calculate_induced_field(
        extend_N_cd=128 * np.ones(shape=(3, 2), dtype=int),
        deextend=True,
    )
    ind_g = ind.Ffe_wg[0]

    # Save as a cube file
    big_g = gd.collect(ind_g)
    if world.rank == 0:
        write(f'fe_{freq.freq * au_to_eV:.2f}.cube', calc.atoms, data=big_g)


do(0)
do(1)
