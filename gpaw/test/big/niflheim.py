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
        if job.ncpus % 24 == 0:
            partition = 'xeon24'
            size = 24
        else:
            partition = 'xeon8'
            size = 8

        cmd = ['sbatch',
               '--partition={}'.format(partition),
               '--job-name={}'.format(job.name),
               '--time={}'.format(job.walltime // 60),
               '--ntasks={}'.format(job.ncpus)]

        if job.ncpus % size == 0:
            cmd.append('--nodes={}'.format(job.ncpus // size))

        mpi_cmd = 'mpirun '
        if partition == 'xeon24':
            mpi_cmd += '-mca pml cm -mca mtl psm2 -x OMP_NUM_THREADS=1 '

        script = [
            '#!/bin/bash -l',
            'touch {}.start'.format(job.name),
            mpi_cmd + 'gpaw-python {}.py {} > {}.output'
            .format(job.script, job.args, job.name),
            'echo $? > {}.done'.format(job.name)]
        if 1:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(('\n'.join(script) + '\n').encode())
            assert p.returncode == 0
        os.chdir(dir)
