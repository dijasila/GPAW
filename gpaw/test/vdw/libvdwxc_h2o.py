from ase.build import molecule
from gpaw import GPAW, Mixer
from gpaw.atom.generator import Generator
from gpaw.xc.libvdwxc import vdw_df, vdw_mbeef

system = molecule('H2O')
system.center(vacuum=1.5)
system.pbc = 1

sol_setups = {}
for sym in 'H', 'O':
    s = Generator(sym, xcname='PBEsol')
    setup = s.run(write_xml=False)
    sol_setups[sym] = setup

for mode in ['lcao', 'fd', 'pw']:
    for vdw in 'df', 'mbeef':
        if mode == 'lcao' and vdw == 'mbeef':
            continue
        print(mode, vdw)
        kwargs = dict(mode=mode,
                      basis='szp(dzp)',
                      mixer=Mixer(0.3, 5, 10.))
        if vdw == 'df':
            kwargs['xc'] = vdw_df()
        elif vdw == 'mbeef':
            kwargs['xc'] = vdw_mbeef()
            kwargs['setups'] = sol_setups
        else:
            assert 0
        calc = GPAW(**kwargs)
        def stopcalc():
            calc.scf.converged = True

        calc.attach(stopcalc, 6)

        system.set_calculator(calc)
        system.get_potential_energy()
