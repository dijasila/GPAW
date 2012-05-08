def agts(queue):
    run = queue.add('ScSZ.py', ncpus=8, walltime=13*60, deps=[])
