import os
import glob
import subprocess

from gpaw.test.big.agts import AGTSQueue
from gpaw.test.big.niflheim import NiflheimCluster


def cmd(c):
    x = os.system(c)
    assert x == 0, c


os.chdir('agts')

cmd('svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw')

cmd("""echo "\
cd agts/gpaw&& \
source /home/camp/modulefiles.sh&& \
module load NUMPY&& \
module load open64/4.2.3-0 && \
python setup.py --remove-default-flags --customize=\
doc/install/Linux/Niflheim/el5-xeon-open64-goto2-1.13-acml-4.4.0.py \
build_ext" | ssh thul bash""")

cmd("""echo "\
cd agts/gpaw&& \
source /home/camp/modulefiles.sh&& \
module load NUMPY&& \
module load open64/4.2.3-0 && \
python setup.py --remove-default-flags --customize=\
doc/install/Linux/Niflheim/el5-opteron-open64-goto2-1.13-acml-4.4.0.py \
build_ext" | ssh fjorm bash""")

cmd("""wget --no-check-certificate --quiet \
http://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-latest.tar.gz && \
tar xzf gpaw-setups-latest.tar.gz && \
rm gpaw-setups-latest.tar.gz && \
mv gpaw-setups-[0-9]* gpaw/gpaw-setups""")

cmd('svn checkout https://svn.fysik.dtu.dk/projects/ase/trunk ase')

queue = AGTSQueue()
queue.collect()
cluster = NiflheimCluster('~/agts/ase', '~/agts/gpaw/gpaw-setups')
#queue.jobs = [job for job in queue.jobs if job.script == 'testsuite.agts.py']
nfailed = queue.run(cluster)

queue.copy_created_files('/home/camp2/jensj/WWW/gpaw-files')

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
            mailto='jensj@fysik.dtu.dk',
            mailserver='servfys.fysik.dtu.dk',
            attachment='status.log')
