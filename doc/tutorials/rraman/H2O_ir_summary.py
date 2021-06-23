# web-page: H2O_ir_summary.txt
from ase import io
from ase.vibrations.infrared import Infrared

atoms = io.read('relaxed.traj')
ir = Infrared(atoms, name='ir')
with open('H2O_ir_summary.txt', 'w') as fd:
    ir.summary(log=fd)
