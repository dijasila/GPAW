# web-page: H2O_Placzek_static_summary.txt
from ase import io
from ase.vibrations.placzek import PlaczekStatic

atoms = io.read('relaxed.traj')
ram = PlaczekStatic(atoms, name='static_raman')
with open('H2O_Placzek_static_summary.txt', 'w') as fd:
    ram.summary(log=fd)
