import ase.db
from ase import Atoms

from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters, parameters_extra
from gpaw.atom.check import check, all_names


con = ase.db.connect('datasets.db')
for name in all_names:
    #check(con, name)

    id = con.reserve(name=name, test='dataset')
    if id is None:
        continue
        
    if '.' in name:
        symbol, e = name.split('.')
        params = parameters_extra[symbol]
        assert params['name'] == e
    else:
        symbol = name
        params = parameters[symbol]
        
    gen = Generator(symbol, 'PBE', scalarrel=True)
    gen.run(write_xml=False, **params)
    nlfer = []
    for j in range(gen.njcore):
        nlfer.append((gen.n_j[j], gen.l_j[j], gen.f_j[j], gen.e_j[j], 0.0))
    for n, l, f, eps in zip(gen.vn_j, gen.vl_j, gen.vf_j, gen.ve_j):
        nlfer.append((n, l, f, eps, gen.rcut_l[l]))
    con.write(Atoms(symbol), test='dataset', name=name, data={'nlfer': nlfer})
    del con[id]
