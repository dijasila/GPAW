import os
import subprocess

from gpaw.test.big.agts import Cluster


class NiflheimCluster(Cluster):
    def __init__(self, asepath='', setuppath='$GPAW_SETUP_PATH'):
        self.asepath = asepath
        self.setuppath = setuppath

    def submit(self, job):
        dir = os.getcwd()
        os.chdir(job.dir)

        self.write_pylab_wrapper(job)

        gpaw_python = os.path.join(dir, 'gpaw', 'build',
                                   'bin.' + '-2.6', 'gpaw-python')

        submit_pythonpath = ':'.join([
            self.asepath,
            dir + '/gpaw',
            '%s/gpaw/build/lib.%s' % (dir, arch),
            '$PYTHONPATH'])

        cmd = ['sbatch',
               '--job-name={}'.format(job.name)]

        script = [
            '#!/bin/sh'
            'touch {}.start'.format(job.name),
            'mpirun -x PYTHONPATH={} \\'.format(submit_pythonpath),
            '       -x GPAW_SETUP_PATH={} \\'.format(self.setuppath),
            '       -x OMP_NUM_THREADS=1 \\'
            '{} {}.py {} > {}.output'.format(
                gpaw_python, job.script, job.args, job.name),
            'echo $? > {}.done'.format(job.name)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate(script)
        assert p.returncode == 0
        os.chdir(dir)
