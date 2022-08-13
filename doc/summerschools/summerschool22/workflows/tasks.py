def relax(atoms, calculator):
    from ase.build import bulk
    from ase.constraints import ExpCellFilter
    from ase.optimize import BFGS
    from gpaw import GPAW
    atoms.calc = GPAW(**calculator)
    opt = BFGS(ExpCellFilter(atoms), trajectory='opt.traj',
               logfile='opt.log')
    opt.run(fmax=0.01)
    # Remove the calculator, since we only want to return
    # the atoms to ASR:
    atoms.calc = None
    return atoms


def groundstate(atoms, calculator):
    from pathlib import Path
    from gpaw import GPAW
    atoms.calc = GPAW(**calculator)
    atoms.get_potential_energy()
    path = Path('groundstate.gpw')
    atoms.calc.write(path)
    return path


def bandstructure(gpw):
    from gpaw import GPAW
    gscalc = GPAW(gpw)
    atoms = gscalc.get_atoms()
    bandpath = atoms.cell.bandpath(npoints=50)
    bandpath.write('bandpath.json')
    gscalc.fixed_density(kpts=bandpath, symmetry='off')
    bs = gscalc.band_structure()
    bs.write('bs.json')
    return bs
