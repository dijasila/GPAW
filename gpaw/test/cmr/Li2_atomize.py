import os

import cmr
# set True in order to use cmr in parallel jobs!
cmr.set_ase_parallel(enable=True)

from ase.structure import molecule
from ase.io import read, write
from ase.parallel import rank

from gpaw import GPAW, restart

calculate = True
recalculate = True
analyse_from_dir = True # analyse local cmr files

upload_to_db = False  # upload cmr files to the database
analyse_from_db = False # analyse database

create_group = True # group calculations beloging to a given reaction

clean = True

if create_group: assert analyse_from_dir or analyse_from_db

if analyse_from_db: assert upload_to_db

# define the project in order to find it in the database!
project_id = 'my first project: Li2 atomize'

vacuum = 3.5

# calculator parameters
xc = 'LDA'
mode = 'lcao'
h = 0.20

cmr_params_template = {
    'db_keywords': [project_id],
    # add project_id also as a field to support search across projects
    'project_id': project_id,
    # user's tags
    'U_vacuum': vacuum,
    'U_xc': xc,
    'U_mode': mode,
    'U_h': h,
    }

if calculate:

    # molecule
    formula = 'Li2'
    # set formula name to be written into the cmr file
    cmr_params = cmr_params_template.copy()
    cmr_params['U_formula'] = formula

    cmrfile = formula + '.cmr'

    system = molecule(formula)
    system.center(vacuum=vacuum)
    # Note: Molecules do not need broken cell symmetry!
    if 0:
        system.cell[1, 1] += 0.01
        system.cell[2, 2] += 0.02

    # Hund rule (for atoms)
    hund = (len(system) == 1)
    cmr_params['U_hund'] = hund

    # first calculation: LDA lcao
    calc = GPAW(mode=mode, xc=xc, h=h, hund=hund, txt=formula + '.txt')
    system.set_calculator(calc)
    e = system.get_potential_energy()
    # write gpw file
    calc.write(formula)
    # add total energy to users tags
    cmr_params['U_potential_energy'] = e
    # write the information 'as in' corresponding trajectory file
    # plus cmr_params into cmr file
    write(cmrfile, system, cmr_params=cmr_params)

    del calc

    # atom
    formula = 'Li'
    # set formula name to be written into the cmr file
    cmr_params = cmr_params_template.copy()
    cmr_params['U_formula'] = formula

    cmrfile = formula + '.cmr'

    system = molecule(formula)
    system.center(vacuum=vacuum)
    # Note: Li does not need broken cell symmetry! Many other atoms do!
    if 0:
        system.cell[1, 1] += 0.01
        system.cell[2, 2] += 0.02

    # Hund rule (for atoms)
    hund = (len(system) == 1)
    cmr_params['U_hund'] = hund

    # first calculation: LDA lcao
    calc = GPAW(mode=mode, xc=xc, h=h, hund=hund, txt=formula + '.txt')
    system.set_calculator(calc)
    e = system.get_potential_energy()
    # write gpw file
    calc.write(formula)
    # add total energy to users tags
    cmr_params['U_potential_energy'] = e
    # write the information 'as in' corresponding trajectory file
    # plus cmr_params into cmr file
    write(cmrfile, system, cmr_params=cmr_params)

    del calc

if recalculate:

    # now calculate PBE energies on LDA orbitals

    # molecule
    formula = 'Li2'
    system, calc = restart(formula, txt=None)

    ediff = calc.get_xc_difference('PBE')

    cmrfile = formula + '.cmr'

    # add new results to the cmrfile
    assert os.path.exists(cmrfile)
    data = cmr.read(cmrfile)
    data.set_user_variable('U_potential_energy_PBE', data['U_potential_energy'] + ediff)
    data.write(cmrfile)

    del calc

    # atom
    formula = 'Li'
    system, calc = restart(formula, txt=None)

    ediff = calc.get_xc_difference('PBE')

    cmrfile = formula + '.cmr'

    # add new results to the cmrfile
    assert os.path.exists(cmrfile)
    data = cmr.read(cmrfile)
    data.set_user_variable('U_potential_energy_PBE', data['U_potential_energy'] + ediff)
    data.write(cmrfile)

    del calc

if analyse_from_dir:

    # analyse the results from cmr files in the local directory

    from cmr.ui import DirectoryReader

    reader = DirectoryReader(directory='.', ext='.cmr')
    # read all compounds in the project with lcao and LDA orbitals
    all = reader.find(name_value_list=[('U_mode', 'lcao'), ('U_xc', 'LDA')],
                      keyword_list=[project_id])

    if rank == 0:
        print 'results from cmr files in the local directory'
    # print requested results
    # column_length=0 aligns data in the table (-1 : data unaligned is default)
    all.print_table(column_length=0,
                    columns=['U_formula', 'U_vacuum',
                             'U_xc', 'U_h', 'U_hund',
                             'U_potential_energy', 'U_potential_energy_PBE',
                             'ase_temperature'])

    # access the results directly and calculate atomization energies
    f2 = 'Li2'
    f1 = 'Li'

    if rank == 0:

        # results are accesible only on master rank

        r2 = all.get('U_formula', f2)
        r1 = all.get('U_formula', f1)

        ea_LDA = 2 * r1['U_potential_energy'] - r2['U_potential_energy']
        ea_PBE = 2 * r1['U_potential_energy_PBE'] - r2['U_potential_energy_PBE']

        print 'atomization energy [eV] ' + xc + ' = ' + str(ea_LDA)
        print 'atomization energy [eV] PBE = ' + str(ea_PBE)

        if create_group:
            # ea_LDA and ea_PBE define a group
            group = cmr.create_group();
            group.add(r1['db_hash']);
            group.add(r2['db_hash']);
            group.set_user_variable('U_ea_LDA', ea_LDA)
            group.set_user_variable('U_ea_PBE', ea_PBE)
            group.set_user_variable('U_description', 'atomization energy [eV]')
            group.set_user_variable('U_reaction', '2 * Li - Li2')
            group.set_user_variable('db_keywords', [project_id])
            group.set_user_variable('project_id', project_id)
            group.write("Li2_atomize_from_dir.cmr");

if upload_to_db:

    # upload cmr files to the database

    if rank == 0:
        os.system("cmr --commit Li*.cmr")

if analyse_from_db:

    # analyse the results from the database

    from cmr.ui import DBReader
    reader = DBReader()
    all = reader.find(name_value_list=[('U_mode', 'lcao'),
                                       ('U_xc', 'LDA'),
                                       #('db_user', '')
                                       ],
                      keyword_list=[project_id])

    if rank == 0:
        print 'results from the database'
    # print requested results
    # column_length=0 aligns data in the table (-1 : data unaligned is default)
    all.print_table(column_length=0,
                    columns=['U_formula', 'U_vacuum',
                             'U_xc', 'U_h', 'U_hund',
                             'U_potential_energy', 'U_potential_energy_PBE',
                             'ase_temperature'])

    # access the results directly and calculate atomization energies
    f2 = 'Li2'
    f1 = 'Li'

    if rank == 0:

        # results are accesible only on master rank

        r2 = all.get('U_formula', f2)
        r1 = all.get('U_formula', f1)

        ea_LDA = 2 * r1['U_potential_energy'] - r2['U_potential_energy']
        ea_PBE = 2 * r1['U_potential_energy_PBE'] - r2['U_potential_energy_PBE']

        print 'atomization energy [eV] ' + xc + ' = ' + str(ea_LDA)
        print 'atomization energy [eV] PBE = ' + str(ea_PBE)

        if create_group:
            # ea_LDA and ea_PBE define a group
            group = cmr.create_group();
            group.add(r1['db_hash']);
            group.add(r2['db_hash']);
            group.set_user_variable('U_ea_LDA', ea_LDA)
            group.set_user_variable('U_ea_PBE', ea_PBE)
            group.set_user_variable('U_description', 'atomization energy [eV] (created directly from database)')
            group.set_user_variable('U_reaction', '2 * Li - Li2')
            group.set_user_variable('db_keywords', [project_id])
            group.set_user_variable('project_id', project_id)
            group.write("Li2_atomize_from_db.cmr");
            group.write(".cmr");

if clean:

    if rank == 0:
        for file in ['Li.cmr', 'Li.gpw', 'Li.txt', 'Li2.cmr', 'Li2.gpw', 'Li2.txt',
                     "Li2_atomize_from_dir.cmr", "Li2_atomize_from_db.cmr"]:
            if os.path.exists(file): os.unlink(file)
