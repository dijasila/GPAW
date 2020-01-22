import numpy as np
from gpaw.defects import ElectrostaticCorrections

FWHM = 2.0
q = -3
epsilon = 12.7
formation_energies = []
repeats = [1, 2, 3, 4]
for repeat in repeats:
    pristine = 'GaAs{0}{0}{0}.pristine.gpw'.format(repeat)
    defect = 'GaAs{0}{0}{0}.Ga_vac.gpw'.format(repeat)
    elc = ElectrostaticCorrections(pristine=pristine,
                                   defect=defect,
                                   q=q,
                                   FWHM=FWHM)
    electrostatic_data = elc.calculate_potentials(epsilon)
    formation_energies.append(elc.calculate_formation_energies(epsilon))
    electrostatic_data['El'] = elc.El
    np.savez('electrostatic_data_{0}{0}{0}.npz'.format(repeat),
             **electrostatic_data)

formation_energies = np.array(formation_energies)
uncorrected = formation_energies[:, 0]
corrected = formation_energies[:, 1]
np.savez('formation_energies.npz',
         repeats=np.array(repeats),
         corrected=corrected,
         uncorrected=uncorrected)
