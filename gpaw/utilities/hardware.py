import os
import re

moab = {
    'cmdstr': '#MSUB ',
    'jobid': '$MOAB_JOBID',
    'mailtype': '-m bea',
    'mpirun': 'mpiexec -np ',
    'name': '-N ',
    'nodes': '-l nodes=',
    'ppn': ':ppn=',
    'walltime': '-l walltime=',
}
sbatch = {
    'cmdstr': '#SBATCH ',
    'jobid': '$SLURM_JOB_ID',
    'mailtype': '--mail-type=ALL',
    'mpirun': 'mpiexec -np ',
    'name': '--job-name=',
    'nodes': '--nodes=',
    'tasks': '--tasks-per-node=',
    'walltime': '--time=',
}

_hardware_info = {
    "bwUniCluster": {
        "cores_per_node": 16,
        "loginnodes": [r'uc1n*'],
        'scheduler': moab,
    },
    "jureca": {
        "cores_per_node": 24,
        "loginnodes": ["jureca"],
        'scheduler': {
            'cmdstr': '#SBATCH ',
            'jobid': '$SLURM_JOBID',
            'mpirun': 'srun -n ',
            'mail': '--mail-user=',
            'mailtype': '--mail-type=ALL',
            'name': '--job-name=',
            'nodes': '--nodes=',
            'walltime': '--time=',
        },
    },
    "nemo": {
        "cores_per_node": 20,
        "loginnodes": [r"login1.nemo.privat"],
        'scheduler': moab,
    },
    "justus2": {
        "cores_per_node": 48,
        "loginnodes": [r"login??"],
        'scheduler': sbatch,
    }
}


def dhms(secs):
    """return days,hours,minutes and seconds"""
    dhms = [0, 0, 0, 0]
    dhms[0] = int(secs // 86400)
    s = secs % 86400
    dhms[1] = int(s // 3600)
    s = secs % 3600
    dhms[2] = int(s // 60)
    s = secs % 60
    dhms[3] = int(s + .5)
    return dhms


def hms(secs):
    """return hours,minutes and seconds"""
    hms = [0, 0, 0]
    hms[0] = int(secs // 3600)
    s = secs % 3600
    hms[1] = int(s // 60)
    s = secs % 60
    hms[2] = int(s + 0.5)
    return hms


def hms_string(secs):
    """return hours,minutes and seconds string, e.g. 02:00:45"""
    l = hms(secs)

    def extend10(n):
        if n < 10:
            return '0' + str(n)
        else:
            return str(n)

    return extend10(l[0]) + ':' + extend10(l[1]) + ':' + extend10(l[2])


class ComputeCluster:
    def __init__(self, architecture=None):
        if architecture:
            self.arch = architecture
            try:
                self.data = _hardware_info[architecture]
                return
            except KeyError:
                raise KeyError(
                    'Architecture {} unknown, known are\n'.format(
                        architecture) + self.list_architectures())

        def get_hostname():
            if os.path.isfile('/etc/FZJ/systemname'):
                with open('/etc/FZJ/systemname') as f:
                    return f.read().strip()

            if 'HOSTNAME' in list(os.environ.keys()):
                return os.environ['HOSTNAME']

            try:
                import socket
                return socket.gethostname().split('-')[0]
            except Exception:
                dummy, hostname = os.popen4('hostname -s')
                return hostname.readline().split()

        def has_key_regexp(dictionary, expression):
            for key in dictionary:
                if re.match(key, expression):
                    return True
            return False

        hostname = get_hostname()
        for host in _hardware_info:
            d = _hardware_info[host]
            if has_key_regexp(d['loginnodes'], hostname):
                self.arch = host
                self.data = d
                return
        raise KeyError(f'Host {hostname} unknown, try -a option.\n' +
                       self.list_architectures())

    def list_architectures(self):
        string = ''
        for arch in _hardware_info:
            string += f'  {arch}\n'
        return string

    def write(self, filename=None, **set):
        if filename is None:
            filename = 'run.' + self.arch
        f = open(filename, 'w')

        env = os.environ
        d = self.data['scheduler']
        c = d['cmdstr']

        print('#!/bin/bash -x', file=f)

        print(c + d['name'] + set['name'].replace('+', ''), file=f)
        cores = set['cores']
        cores_per_node = self.data['cores_per_node']
        if set['smt']:
            cores_per_node *= 2
        nodes = int((cores + (cores_per_node - 1)) / cores_per_node)
        ppn = int((cores + nodes - 1) / nodes)
        if cores != nodes * ppn:
            print('Note:', nodes * ppn, 'cores reserved but only', cores,
                  'cores used.')
            print('     Consider to use multiples of', cores_per_node, end=' ')
            print('processors for best performance.')
        print(c + d['nodes'] + str(nodes), file=f, end='')
        if 'ppn' in d:
            print(d['ppn'] + str(ppn), file=f)
        else:
            print(file=f)
            print(c + d['tasks'] + str(ppn), file=f)
        print(c + d['walltime'] + hms_string(set['time']), file=f)
        if set['mail'] is not None:
            print(c + '--mail-user=' + set['mail'], file=f)
        print(c + d['mailtype'], file=f)
        print('cd', set['wd'], file=f)
        print('export MODULEPATH=' + env['MODULEPATH'], file=f)
        for module in env['LOADEDMODULES'].split(':'):
            print('module load', module, file=f)
        print(d['mpirun'] + str(cores) + ' gpaw python',
              set['script'], end=' ', file=f)
        if 'parameters' in set:
            print(set['parameters'], end=' ', file=f)
        print('>', set['out'] + '_' + d['jobid'], end=' ', file=f)
        print('2>', set['err'] + '_' + d['jobid'], file=f)
        f.close()

        return filename
