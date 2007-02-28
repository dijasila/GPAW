import os
import sys


def get_mpi_implementation():
    mpi = os.environ.get('GPAW_MPI_IMPLEMENTATION')
    if mpi is not None:
        return mpi
    
    machine = os.uname()[4]
    
    if machine == 'sun4u':
        return 'sun'

    if sys.platform == 'aix5':
        return 'poe'

    if sys.platform == 'ia64':
        return 'mpich'
                
    output = os.popen3('mpicc --showme')[1].read()
    if output != '':
        if 'openmpi' in output:
            return 'openmpi'
        else:
            return 'lam'

    output = os.popen3('mpicc -show')[1].read()
    if output != '':
        if 'mvapich' in output:
            return 'mvapich'
        else:
            return 'mpich'

    return ''


def get_mpi_command(debug=False):
    cmd = os.environ.get('GPAW_MPI_COMMAND')
    if cmd is not None:
        return cmd

    mpi = get_mpi_implementation()

    if mpi == 'sun':
        return 'mprun -np %(np)d %(job)s &'

    if mpi == 'poe':
        if os.environ.has_key('LOADL_PROCESSOR_LIST'):
            return "poe '%(job)s' &"
        else:
            return "poe '%(job)s' -procs %(np)d -hfile %(hostfile)s &"
                
    if mpi == 'mpich':
        output = os.popen3('mpiexec')[1].read()
        if output != '':
            return 'mpiexec -n %(np)d %(job)s'
        output = os.popen3('mpirun')[1].read()
        if output != '':
            return 'mpirun -n %(np)d %(job)s'
        raise NotImplementedError

    if mpi == 'mvapich':
        raise NotImplementedError

    if mpi == 'lam':
        if debug:
            return 'lamboot -v %(hostfile)s; mpirun -v -np %(np)d %(job)s &'
        else:
            return 'lamboot -H %(hostfile)s; mpirun -np %(np)d %(job)s &'

    if mpi == 'openmpi':
        return 'mpirun -np %(np)d --hostfile %(hostfile)s %(job)s &'

    raise NotImplementedError


if __name__ == '__main__':
    print get_mpi_implementation()
    print get_mpi_command(debug=False)
    print get_mpi_command(debug=True)

