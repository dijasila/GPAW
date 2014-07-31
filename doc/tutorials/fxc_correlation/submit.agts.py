def agts(queue):
    queue.add('hydrogen.py', ncpus=1, walltime=120)
    gs_CO = queue.add('gs_CO.py', ncpus=1, walltime=1000)
    gs_diamond = queue.add('gs_diamond.py', ncpus=1, walltime=100)
    queue.add('rapbe_CO.py', deps=gs_CO, ncpus=16, walltime=1200)
    queue.add('rapbe_diamond.py', deps=gs_diamond, ncpus=16, walltime=1200)
