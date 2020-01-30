from gpaw.atom.generator import Generator
from gpaw.atom.basis import BasisMaker
from gpaw.atom.configurations import parameters, parameters_extra

xc = 'GLLBSC'
name = 'my'
if 'Ag' in parameters_extra:
    args = parameters_extra['Ag']  # Choose the 11-electron setup
else:
    args = parameters['Ag']  # Choose the 17-electron setup
args.update(dict(name=name, use_restart_file=False, exx=True))

# Generate setup
generator = Generator('Ag', xc, scalarrel=True)
generator.run(write_xml=True, **args)

# Generate basis
bm = BasisMaker('Ag', name='{}.{}'.format(name, xc), xc=xc, run=False)
bm.generator.run(write_xml=False, **args)
basis = bm.generate(zetacount=2, polarizationcount=0,
                    jvalues=[0, 1, 2],  # include d, s and p
                    )
basis.write_xml()
