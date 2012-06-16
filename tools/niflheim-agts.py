import os
import sys
import glob
import shutil
import subprocess

def cmd(c):
    x = os.system(c)
    assert x == 0, c

def fail(subject, email=None, filename='/dev/null'):
    import os
    if email is not None:
        assert os.system('uuencode %s %s | mail -s "%s" %s' %
                         (filename, os.path.basename(filename),
                          subject, email)) == 0
    raise SystemExit

if '--dir' in sys.argv:
    i = sys.argv.index('--dir')
    dir = sys.argv[i+1]
else:
    dir = 'agts'

if '--email' in sys.argv:
    i = sys.argv.index('--email')
    email = sys.argv[i+1]
else:
    email = None

assert os.path.isdir(dir)

gpawdir = os.path.join(dir, 'gpaw')

# remove the old run directory
if os.path.isdir(dir):
    shutil.rmtree(dir)

os.mkdir(dir)
os.chdir(dir)

cmd('svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw')

# a version of gpaw is needed for imports from within this script!
cmd("\
cd " + gpawdir + "&& \
source /home/camp/modulefiles.sh&& \
module load NUMPY&& \
python setup.py build_ext 2>&1 > build_ext.log")

# import gpaw from where it was installed
sys.path.insert(0, gpawdir)

cmd("echo '\
cd '" + gpawdir + "'&& \
source /home/camp/modulefiles.sh&& \
module load NUMPY&& \
module load open64/4.2.3-0 && \
module load openmpi/1.3.3-1.el5.fys.open64.4.2.3 && \
module load hdf5/1.8.6-5.el5.fys.open64.4.2.3.openmpi.1.3.3 && \
python setup.py --remove-default-flags --customize=\
doc/install/Linux/Niflheim/el5-xeon-open64-acml-4.4.0-acml-4.4.0-hdf-SL-2.0.1.py \
build_ext 2>&1 > thul.log' | ssh thul bash")

cmd("echo '\
cd '" + gpawdir + "'&& \
source /home/camp/modulefiles.sh&& \
module load NUMPY&& \
module load open64/4.2.3-0 && \
python setup.py --remove-default-flags --customize=\
doc/install/Linux/Niflheim/el5-opteron-open64-acml-4.4.0-acml-4.4.0-hdf-SL-2.0.1.py \
build_ext 2>&1 > fjorm.log' | ssh fjorm bash")

cmd("""wget --no-check-certificate --quiet \
http://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-latest.tar.gz && \
tar xzf gpaw-setups-latest.tar.gz && \
rm gpaw-setups-latest.tar.gz && \
mv gpaw-setups-[0-9]* gpaw/gpaw-setups""")

cmd('svn export https://svn.fysik.dtu.dk/projects/ase/trunk ase')

# ase needed
sys.path.insert(0, '%s/ase' % dir)

from gpaw.test.big.agts import AGTSQueue
from gpaw.test.big.niflheim import NiflheimCluster

queue = AGTSQueue()
queue.collect()
cluster = NiflheimCluster(asepath=os.path.join(dir, 'ase'),
                          setuppath=os.path.join(gpawdir, 'gpaw-setups'))
# Example below is confusing: job.script must NOT be the *.agts.py script,
# but the actual python script to be run!
# testsuite.agts.py does both: see gpaw/test/big/miscellaneous/testsuite.agts.py
#queue.jobs = [job for job in queue.jobs if job.script == 'testsuite.agts.py']

nfailed = queue.run(cluster)

gfiles = os.path.join(dir, 'gpaw-files')
if not os.path.isdir(gfiles):
    os.mkdir(gfiles)

queue.copy_created_files(gfiles)

# make files readable by go
files = glob.glob(gfiles + '/*')
for f in files:
    os.chmod(f, 0644)

from gpaw.version import version

subject = 'AGTS GPAW %s: ' % str(version)
# Send mail:
if nfailed:
    subject += ' failed'
    fail(subject, email, os.path.join(dir, 'status.log'))
else:
    subject += ' succeeded'
    fail(subject, email, os.path.join(dir, 'status.log'))

if 0:
    # Analysis:
    import matplotlib
    matplotlib.use('Agg')
    from gpaw.test.big.analysis import analyse
    user = os.environ['USER']
    analyse(queue,
            '../analysis/analyse.pickle',  # file keeping history
            '../analysis',                 # Where to dump figures
            rev=niflheim.revision,
            #mailto='gpaw-developers@listserv.fysik.dtu.dk',
            mailserver='servfys.fysik.dtu.dk',
            attachment='status.log')
