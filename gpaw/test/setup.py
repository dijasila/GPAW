import sys

from gpaw.commands.cli import run


names = sys.argv[1:]

for collection, ecut, scale in [('fcc', 400, 1),
                                ('fcc', 600, 1),
                                ('fcc', 800, 1),
                                ('fcc', 800, 0.85),
                                #('rocksalt', 800, 1),
                                ]:#('rocksalt', 800, 0.85)]:
    parameters = 'mode=PW(%f),kpts=2' % ecut
    run('run', names=names, collection='gpaw.collections.' + collection,
        parameters=parameters, database='test.json',
        tag='%s-%d-%d' % (collection, ecut, 100 * scale),
        modify='atoms.cell*=%f' % scale, use_lock_file=True, skip=True)

run('egg-box-test', names=names, parameters='xc=PBE', database='test.json',
    grid_spacings='0.18', tag='egg-box-test', use_lock_file=True, skip=True)
