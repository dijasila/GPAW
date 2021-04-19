from ase import io
from ase.vibrations.infrared import Infrared

atoms = io.read('relaxed.traj')
ir = Infrared(atoms, name='ir')
ir.summary()
