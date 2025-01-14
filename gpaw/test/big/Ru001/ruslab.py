from sys import argv
from ase.build import hcp0001, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize.lbfgs import LBFGS
from gpaw import GPAW, Mixer, FermiDirac

tag = 'Ru001'

adsorbate_heights = {'H': 1.0, 'N': 1.108, 'O': 1.257}

slab = hcp0001('Ru', size=(2, 2, 4), a=2.72, c=1.58 * 2.72, vacuum=7.0,
               orthogonal=True)
slab.center(axis=2)

if len(argv) > 1:
    adsorbate = argv[1]
    tag = adsorbate + tag
    add_adsorbate(slab, adsorbate, adsorbate_heights[adsorbate], 'hcp')

slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 3))

calc = GPAW(mode='fd',
            xc='PBE',
            h=0.2,
            mixer=Mixer(0.1, 5, weight=100.0),
            occupations=FermiDirac(width=0.1),
            kpts=[4, 4, 1],
            eigensolver='cg',
            txt=tag + '.txt')
slab.calc = calc

opt = LBFGS(slab, logfile=tag + '.log', trajectory=tag + '.traj')
opt.run(fmax=0.05)
calc.write(tag + '.gpw')
