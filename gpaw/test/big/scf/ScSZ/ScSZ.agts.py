def agts(queue):
    run = queue.add('ScSZ.py', ncpus=8, walltime=8*60, deps=[])
