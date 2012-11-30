def agts(queue):
    run = queue.add('b256H2O.py', ncpus=16, walltime=8*60, deps=[])
