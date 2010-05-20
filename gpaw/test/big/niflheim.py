import os
import glob
import subprocess

from gpaw.test.big.agts import Cluster


class Niflheim(Cluster):

    gpawrepo = 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'
    aserepo = 'https://svn.fysik.dtu.dk/projects/ase/trunk'

    def __init__(self):
        self.dir = os.getcwd()
        self.revision = None

    def install_gpaw(self):
        if os.system('svn checkout %s gpaw' % self.gpawrepo) != 0:
            raise RuntimeError('Checkout of GPAW failed!')

        p = subprocess.Popen(['svnversion', 'gpaw'], stdout=subprocess.PIPE)
        self.revision = int(p.stdout.read())

        if os.system('cd gpaw&& ' +
                     'source /home/camp/modulefiles.sh&& ' +
                     'module load NUMPY&& '+
                     'python setup.py --remove-default-flags ' +
                     '--customize=doc/install/Linux/Niflheim/' +
                     'el5-xeon-gcc43-acml-4.3.0.py ' +
                     'install --home=..') != 0:
            raise RuntimeError('Installation of GPAW failed!')

        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        os.system('mv gpaw-setups-[0-9]* gpaw-setups')

    def install_ase(self):
        if os.system('svn checkout %s ase' % self.aserepo) != 0:
            raise RuntimeError('Checkout of ASE failed!')
        if os.system('cd ase; python setup.py install --home=..'):
            raise RuntimeError('Installation of ASE failed!')

    def install(self):
        self.install_gpaw()
        self.install_ase()

    def submit(self, job):
        dir = os.getcwd()
        os.chdir(job.dir)

        self.write_pylab_wrapper(job)

        gpaw_python = os.path.join(self.dir, 'bin/gpaw-python')

        submit_pythonpath = 'PYTHONPATH=%s/lib/python:%s/lib64/python:$PYTHONPATH ' % (self.dir, self.dir)
        submit_gpaw_setup_path = 'GPAW_SETUP_PATH=%s/gpaw-setups ' % self.dir

        run_command = '. /home/camp/modulefiles.sh&& '
        run_command += 'module load MATPLOTLIB&& ' # loads numpy, matplotlib, ...

        if job.ncpus == 1:
            # don't use mpi here,
            # this allows one to start mpi inside the *.agts.py script
            run_command += ' ' + submit_pythonpath
            run_command += ' ' + submit_gpaw_setup_path
        else:
            run_command += 'module load openmpi/1.3.3-1.el5.fys.gfortran43.4.3.2&& '
            run_command += 'mpiexec --mca mpi_paffinity_alone 1 '
            run_command += '-x ' + submit_pythonpath
            run_command += '-x ' + submit_gpaw_setup_path

        if job.queueopts is None:
            if job.ncpus == 1:
                ppn = '8:xeon5570'
                nodes = 1
            else:
                assert job.ncpus % 8 == 0
                ppn = '8:xeon5570'
                nodes = job.ncpus // 8
            queueopts = '-l nodes=%d:ppn=%s' % (nodes, ppn)
        else:
            queueopts = job.queueopts

        p = subprocess.Popen(
            ['/usr/local/bin/qsub',
             '-V',
             '%s' % queueopts,
             '-l',
             'walltime=%d:%02d:00' %
             (job.walltime // 3600, job.walltime % 3600 // 60),
             '-N',
             job.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p.stdin.write(
            'touch %s.start\n' % job.name +
            run_command +
            '%s %s.py %s > %s.output\n' %
            (gpaw_python, job.script, job.args, job.name) +
            'echo $? > %s.done\n' % job.name)
        p.stdin.close()
        id = p.stdout.readline().split('.')[0]
        job.pbsid = id
        os.chdir(dir)


if __name__ == '__main__':
    from gpaw.test.big.agts import AGTSQueue

    os.chdir(os.path.join(os.environ['HOME'], 'weekend-tests'))

    niflheim = Niflheim()
    if not os.path.isfile('bin/gpaw-python'):
        niflheim.install()

    os.chdir('gpaw')
    queue = AGTSQueue()
    queue.collect()

    selected_queue_jobs = []
    # examples of selecting jobs
    if 0:
        for j in queue.jobs:
            if (
                j.walltime < 3*60 or
                j.dir.startswith('doc') or
                j.dir.startswith('gpaw/test/big/bader_water') or
                j.dir.startswith('doc/devel/memory_bandwidth')
                ):
                selected_queue_jobs.append(j)

    if len(selected_queue_jobs) > 0: queue.jobs = selected_queue_jobs

    nfailed = queue.run(niflheim)

    queue.copy_created_files('/home/camp2/jensj/WWW/gpaw-files')

    # Analysis:
    import matplotlib
    matplotlib.use('Agg')
    from gpaw.test.big.analysis import analyse
    user = os.environ['USER']
    analyse(queue,
            '../analysis/analyse.pickle',  # file keeping history
            '../analysis',                 # Where to dump figures
            rev=niflheim.revision,
            mailto=user,
            mailserver='servfys.fysik.dtu.dk',
            attachment='status.log')

    if nfailed == 0:
        tag = 'success'
    else:
        tag = 'failed'

    os.chdir('..')
    dir = os.path.join('/scratch', user, 'gpaw-' + tag)
    os.system('rm -rf %s-old' % dir)
    os.system('mv %s %s-old' % (dir, dir))
    #os.system('mv gpaw %s' % dir)
