# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
from os.path import dirname, isfile, join
from distutils.util import get_platform
import cPickle as pickle
import socket

from gpaw import debug, trace
from gpaw.utilities.socket import send, recv
from gpaw.mpi.config import get_mpi_command
import gpaw.utilities.timing as timing


class MPIPAW:
    # List of methods for Paw object:
    paw_methods = 0#???dir(Paw)
    
    def __init__(self, **kwargs):
        # Make connection:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        port = 17000
        host = socket.gethostname()
        while True:
            try:
                s.bind((host, port))
            except socket.error:
                port += 1
            else:
                break

        s.listen(1)

        # If the environment variable GPAW_PYTHON is set, use that:
        if 'GPAW_PYTHON' in os.environ:
            gpaw_python = os.environ['GPAW_PYTHON']
        else:
            gpaw_python = None
            # Look in $PATH:
            paths = os.environ.get('PATH').split(os.pathsep)
            # Check build directory first:
            dir = join(dirname(__file__), '../../build',
                       'bin.%s-%s' % (get_platform(), sys.version[0:3]))
            paths = [dir] + paths
            print paths
            for path in paths:
                print path
                if isfile(join(path, 'gpaw-python')):
                    gpaw_python = join(path, 'gpaw-python')
                    break

        if gpaw_python is None:
            raise RuntimeError('Custom interpreter is not found')
        
        # This is the Python command that all processors wil run:
        line = "from gpaw.mpi.run import run; run('%s', %d)" % (host, port)

        options = ''
        if debug:
            options = ' --gpaw-debug'
        if trace:
            options += ' --gpaw-trace'

        # Get the command to start mpi.  Typically this will be
        # something like:
        #
        #   cmd = 'mpirun -np %(np)d --hostfile %(hostfile)s %(job)s &'
        #
        cmd = get_mpi_command(debug)
        try:
            np = len(open(hostfile).readlines())
        except:
            np = None

        # Insert np, hostfile and job:
        cmd = cmd % {'job': job,
                     'hostfile': hostfile,
                     'np': np}

        error = os.system(cmd)
        if error != 0:
            raise RuntimeError

        self.sckt, addr = s.accept()
        
        string = pickle.dumps(args, -1)

        send(self.sckt, string)
        ack = recv(self.sckt)
        if ack != 'Got your arguments - now give me some commands':
            raise RuntimeError
        s.close()

    def stop(self):
        send(self.sckt, pickle.dumps(("Stop", (), {}), -1))
        string = recv(self.sckt)
        self.out.write(string)
        send(self.sckt,
             'Got your output - now send me your CPU time')
        cputime = pickle.loads(recv(self.sckt))
        return cputime

    def __del__(self):
        self.sckt.close()

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        print attr;asdfghsdfg
        """Catch calls to methods and attributes."""
        send(self.sckt, attr)
        data = pickle.loads(recv(self.sckt))
        if data != 'this is not a simple object':
            return data

        return self.method

    def method(self, *args, **kwargs):
        """Communicate with remote calculation.

        Method name and arguments are send to the parallel calculation
        and results are received.  Output flushed from the calculation
        is also picked up and passed on."""

        # Send method name and arguments:
        string = pickle.dumps((args, kwargs), -1)
        send(self.sckt, string)
        
        # Wait for result:
        while True:
            tag, stuff = pickle.loads(recv(self.sckt))
            if tag == 'output':
                # This was not the result - the output was flushed:
                self.out.write(stuff)
                self.out.flush()
                timing.update()#????
            elif tag == 'result':
                return stuff
            else:
                raise RuntimeError('Unknown tag: ' + tag)



def get_parallel_environment():
    """Get the hosts for a parallel run from the parallel environment.

    Return value will be a tuple like (number of hosts, name of
    hostfile), or ``None``, if there is no parallel environment.  """
    
    global env
    try:
        return env
    except NameError:
        env = _get_parallel_environment()
        return env
    

def _get_parallel_environment():
    from gpaw import hosts
    if hosts is not None:
        # The hosts have been set by the command line argument
        # --hosts (see __init__.py):
        return (hosts, '')
    
    if os.environ.has_key('PBS_NODEFILE'):
        # This job was submitted to the PBS queing system.  Get
        # the hosts from the PBS_NODEFILE environment variable:
        hosts = os.environ['PBS_NODEFILE']
        if len(open(hosts).readlines()) == 1:
            return None
        else:
            return (None, hosts)

    if os.environ.has_key('NSLOTS'):
        # This job was submitted to the Grid Engine queing system:
        nhosts = int(os.environ['NSLOTS'])
        if nhosts == 1:
            return None
        else:
            return (nhosts, None)

    if os.environ.has_key('LOADL_PROCESSOR_LIST'):
        return (None, None)

    return None




if 0:
    def stop_paw(self):
        """Delete PAW-object."""
        if isinstance(self.paw, MPIPaw):
            # Stop old MPI calculation and get total CPU time for all CPUs:
            self.parallel_cputime += self.paw.stop()
        self.paw = None
        
    def __del__(self):
        """Destructor:  Write timing output before closing."""

        self.stop_paw()
        
        # Get CPU time:
        c = self.parallel_cputime + timing.clock()
                
        if c > 1.0e99:
            print >> self.out, 'cputime : unknown!'
        else:
            print >> self.out, 'cputime : %f' % c

        print >> self.out, 'walltime: %f' % (time.time() - self.t0)
        mr = maxrss()
        if mr > 0:
            def round(x): return int(100*x/1024.**2+.5)/100.
            print >> self.out, 'memory  : '+str(round(maxrss()))+' MB'
        print >> self.out, 'date    :', time.asctime()
