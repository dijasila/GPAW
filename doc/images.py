""" TODO:

1. we should find a good way in which to store files elsewhere than static

2. currently the files that are not generated by weekly tests are copied
   from srcpath. This needs to be documented.

Make sure that downloaded files are copied to build dir on build
This must (probably) be done *after* compilation because otherwise dirs
may not exist.

"""
from __future__ import print_function
try:
    from urllib2 import urlopen, HTTPError
except ImportError:
    from urllib.request import urlopen
    from urllib.error import HTTPError
import os

srcpath = 'http://wiki.fysik.dtu.dk/gpaw-files'
agtspath = 'http://wiki.fysik.dtu.dk'


def get(path, names, target=None, source=None):
    """Get files from web-server.

    Returns True if something new was fetched."""

    if target is None:
        target = path
    if source is None:
        source = srcpath
    got_something = False
    for name in names:
        src = os.path.join(source, path, name)
        dst = os.path.join(target, name)

        if not os.path.isfile(dst):
            print(dst, end=' ')
            try:
                data = urlopen(src).read()
                sink = open(dst, 'wb')
                sink.write(data)
                sink.close()
                print('OK')
                got_something = True
            except HTTPError:
                print('HTTP Error!')
    return got_something


literature = """
askhl_10302_report.pdf  mortensen_gpaw-dev.pdf      rostgaard_master.pdf
askhl_master.pdf        mortensen_mini2003talk.pdf
marco_master.pdf        mortensen_paw.pdf           ss14.pdf
""".split()
get('doc/literature', literature, 'documentation')

# Note: bz-all.png is used both in an exercise and a tutorial.  Therefore
# we put it in the common dir so far, rather than any of the two places
get('.', ['bz-all.png'], 'static')
get('exercises/wavefunctions', ['co_bonding.jpg'])

get('tutorials/H2', ['ensemble.png'])

get('.', ['2sigma.png', 'co_wavefunctions.png'], 'documentation')
get('exercises/lrtddft', ['spectrum.png'])
get('documentation/xc', 'g2test_pbe0.png  g2test_pbe.png  results.png'.split())
get('performance', 'dacapoperf.png  goldwire.png  gridperf.png'.split(),
    'static')

get('tutorials/xas', ['h2o_xas_3.png', 'h2o_xas_4.png',
                      'xas_illustration.png', 'xas_h2o_convergence.png'])

get('bgp', ['bgp_mapping_intranode.png',
            'bgp_mapping1.png',
            'bgp_mapping2.png'], 'platforms/BGP')

# workshop 2013 and 2016 photos:
get('workshop13', ['workshop13_01_33-1.jpg'], 'static')
get('workshop16', ['gpaw2016-photo.jpg'], 'static')

# files from agtspath

scf_conv_eval_stuff = """
scf_g2_1_pbe0_fd_calculator_steps.png
scf_g2_1_pbe0_fd_energy.csv
scf_dcdft_pbe_pw_calculator_steps.png
scf_dcdft_pbe_pw_energy.csv
""".split()

get('agts-files', scf_conv_eval_stuff, target='documentation/scf_conv_eval',
    source=agtspath)

# Warning: for the moment dcdft runs are not run (files are static)!
dcdft_pbe_aims_stuff = """
dcdft_aims.tight.01.16.db.csv
dcdft_aims.tight.01.16.db_raw.csv
dcdft_aims.tight.01.16.db_Delta.txt
""".split()

get('agts-files', dcdft_pbe_aims_stuff, target='setups', source=agtspath)

# Warning: for the moment dcdft runs are not run (files are static)!
dcdft_pbe_gpaw_pw_stuff = """
dcdft_pbe_gpaw_pw.csv
dcdft_pbe_gpaw_pw_raw.csv
dcdft_pbe_gpaw_pw_Delta.txt
""".split()

get('agts-files', dcdft_pbe_gpaw_pw_stuff, target='setups', source=agtspath)

# Warning: for the moment dcdft runs are not run (files are static)!
dcdft_pbe_jacapo_stuff = """
dcdft_pbe_jacapo.csv
dcdft_pbe_jacapo_raw.csv
dcdft_pbe_jacapo_Delta.txt
""".split()

get('agts-files', dcdft_pbe_jacapo_stuff, target='setups', source=agtspath)

# Warning: for the moment dcdft runs are not run (files are static)!
dcdft_pbe_abinit_fhi_stuff = """
dcdft_pbe_abinit_fhi.csv
dcdft_pbe_abinit_fhi_raw.csv
dcdft_pbe_abinit_fhi_Delta.txt
""".split()

get('agts-files', dcdft_pbe_abinit_fhi_stuff, target='setups', source=agtspath)

g2_1_stuff = """
pbe_gpaw_nrel_ea_vs.csv pbe_gpaw_nrel_ea_vs.png
pbe_gpaw_nrel_opt_ea_vs.csv pbe_gpaw_nrel_opt_distance_vs.csv
pbe_nwchem_def2_qzvppd_opt_ea_vs.csv pbe_nwchem_def2_qzvppd_opt_distance_vs.csv
""".split()

get('agts-files', g2_1_stuff, target='setups', source=agtspath)

get('tutorials/wannier90', ['GaAs.png', 'Cu.png', 'Fe.png'])

get('agts-files', ['datasets.json'], 'setups', source=agtspath)

# Carlsberg foundation figure:
get('.', ['carlsberg.png'])


def setup(app):
    # Get png files and other stuff from the AGTS scripts that run
    # every weekend:
    from gpaw.test.big.agts import AGTSQueue
    queue = AGTSQueue()
    queue.collect()
    names = set()
    for job in queue.jobs:
        if not job.creates:
            continue
        for name in job.creates:
            assert name not in names, "Name '%s' clashes!" % name
            names.add(name)
            # the files are saved by the weekly tests under agtspath/agts-files
            # now we are copying them back to their original run directories
            path = os.path.join(job.dir, name)
            if os.path.isfile(path):
                continue
            print(path, 'copied from', agtspath)
            get('agts-files', [name], job.dir, source=agtspath)
