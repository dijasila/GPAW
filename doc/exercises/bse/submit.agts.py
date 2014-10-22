def agts(queue):
    a = queue.add('LiF_gs.py', ncpus=1, walltime=10)
    queue.add('LiF_RPA.py', ncpus=8, walltime=10, deps=a)
    # Someone should fix this:
    # queue.add('LiF_BSE.py', ncpus=24, walltime=5 * 60, deps=a)
