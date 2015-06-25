from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw.atom.basis import BasisMaker
args = {'core': '[Kr]',   'rcut': 2.45}
print "Generating setup with following parameters", args
generator = Generator('Ag', 'GLLBSC')
generator.N *= 2 # Increase grid resolution
generator.run(**args)
bm = BasisMaker(generator, name='GLLBSC', run=False)
basis = bm.generate(zetacount=2, polarizationcount=0,
                    energysplit=0.07,
                    jvalues=[0,1,2], # include d, s and p
                    rcutmax=12.0)
basis.write_xml()
