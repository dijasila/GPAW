from ase.build import molecule
from gpaw import GPAW, Mixer

elpa = 0
experimental = {}
if elpa:
    experimental['elpa'] = {}

atoms = molecule('CH3CH2OH', vacuum=3.0)
#atoms = molecule('H2', vacuum=3.0)
calc = GPAW(mode='lcao', basis='dzp',
            parallel=dict(sl_auto=True),
            #txt=None,
            experimental=experimental,
            mixer=Mixer(0.5, 5, 50.0))
atoms.calc = calc
atoms.get_potential_energy()
