def agts(queue):
    a = queue.add('atomization.py', ncpus=1, walltime=30, 
                  creates=['atomization.txt', 'H.gpw', 'H2.gpw'])
    queue.add('relax.py', deps=a, ncpus=1, walltime=30, creates=['optimization.txt'])
    if 0:  # https://trac.fysik.dtu.dk/projects/gpaw/ticket/250
        queue.add('ensembles.py', deps=a, ncpus=1, 
                  walltime=30, creates=['ensemble_energies.txt', 'ensemble.dat'])
