from ase.build import molecule
from ase.optimize import BFGS
from gpaw import GPAW, Davidson
from gpaw.mixer import MixerDif

for name in ['H2', 'N2', 'O2', 'NO']:
    mol = molecule(name)
    mol.center(vacuum=5.0)
    calc = GPAW(mode='fd',
                xc='PBE',
                h=0.2,
                eigensolver=Davidson(3),
                txt=name + '.txt',
                convergence={'eigenstates': 1e-10})
    if name == 'NO':
        mol.translate((0, 0.1, 0))
        calc = calc.new(mixer=MixerDif(0.05, 5), txt=name + '.txt')
    mol.calc = calc

    opt = BFGS(mol, logfile=name + '.log', trajectory=name + '.traj')
    opt.run(fmax=0.05)
    calc.write(name + '.gpw')
