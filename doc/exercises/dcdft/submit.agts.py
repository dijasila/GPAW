def agts(queue):
    a = queue.add('dcdft_gpaw.py', ncpus=4, walltime=20)
    queue.add('ase-db exercise_dcdft.db name=Ca -c+ecut,kpts,width,x,time,iter',
              deps=a, ncpus=1, walltime=5)

