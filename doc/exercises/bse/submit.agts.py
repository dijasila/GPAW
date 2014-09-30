def agts(queue):
    a = queue.add('LiF_gs.py', ncpus=1, walltime=10)
    b = queue.add('LiF_RPA.py', ncpus=8, walltime=10, deps=a)
    cqueue.add('LiF_BSE.py', ncpus=24, walltime=20, deps=a)

