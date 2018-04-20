from ase.db import connect
from ase.optimize.test.systems import create_database

from gpaw import GPAW


def create_database_gpaw():
    create_database()
    db = connect('systems.db')
    systems = [row.toatoms() for row in db.select()]
    for atoms in systems:
        atoms.calc = GPAW(mode='lcao', txt=None)
        db.write(atoms)


if __name__ == '__main__':
    create_database_gpaw()
