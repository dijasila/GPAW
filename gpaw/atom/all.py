from gpaw.atom.aeatom import AllElectronAtom
from gpaw.lcao.readbasis import build_library

with open('ref','r') as f:
    lines = f.readlines()

basisdata = []
for line in lines:
    name, energy = line.split()
    basisdata.append( (name, float(energy) ) )

f = open('out','w')

library = build_library('/home/niflheim/kuisma/bases/h', format = 'turbomole')
for name, ref_energy in basisdata:
    basis = library.get('h-'+name)[:1] # only s
    assert len(basis)>0
    aea = AllElectronAtom('H', xc='null', ee_interaction=False)
    aea.initialize(override_basis_l = basis,ngpts=60000, alpha2=150000000,rcut=70.0)
    aea.run()
    print(name, ref_energy, aea.etot, aea.etot-ref_energy, file=f)
