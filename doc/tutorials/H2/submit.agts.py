def agts(queue):
    a = queue.add('atomization.py', ncpus=1, walltime=30, 
                  creates=['atomization.txt', 'H.gpw', 'H2.gpw'])
