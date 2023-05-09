from gpaw.lcao.generate_ngto_augmented import \
    create_GTO_dictionary as GTO, generate_nao_ngto_basis

gtos_atom = {'H': [GTO('s', 0.02974),
                   GTO('p', 0.14100)],
             'C': [GTO('s', 0.04690),
                   GTO('p', 0.04041),
                   GTO('d', 0.15100)],
             'O': [GTO('s', 0.07896),
                   GTO('p', 0.06856),
                   GTO('d', 0.33200)],
             }

for atom in ['H', 'C', 'O']:
    generate_nao_ngto_basis(atom, xc='PBE', name='aug',
                            nao='dzp', gtos=gtos_atom[atom])
