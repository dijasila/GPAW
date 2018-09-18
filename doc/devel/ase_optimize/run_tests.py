import ase.db
from ase.optimize.test.test import (test_optimizer, all_optimizers,
                                    get_optimizer)
from gpaw import GPAW


db1 = ase.db.connect('systems.db')
db = ase.db.connect('results-lcao.db')


def lcao(txt):
    return GPAW(mode='lcao',
                basis='dzp',
                kpts={'density': 2.0},
                txt=txt)


systems = [(row.name, row.toatoms()) for row in db1.select()]

for opt in all_optimizers:
    optimizer = get_optimizer(opt)
    test_optimizer(systems, optimizer, lcao, 'lcao-', db)
