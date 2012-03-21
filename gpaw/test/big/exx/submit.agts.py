def agts(queue):
    molecules = [queue.add('molecules.py %d' % i,
                           ncpus=1,
                           walltime=40)
                 for i in range(4)]
    queue.add('check.py', deps=molecules)
 
