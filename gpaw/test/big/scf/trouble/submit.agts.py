def agts(queue):
    run16 = queue.add('run.py 16', ncpus=16, walltime=10 * 60)
    run8a = queue.add('run.py 8a', ncpus=8, walltime=40 * 60)
    run8b = queue.add('run.py 8b', ncpus=8, walltime=40 * 60)
    run4a = queue.add('run.py 4a', ncpus=4, walltime=20 * 60)
    run4b = queue.add('run.py 4b', ncpus=4, walltime=20 * 60)
    run1 = queue.add('run.py 1', ncpus=1, walltime=30)
    queue.add('analyse.py', ncpus=1, walltime=10,
              deps=[run16, run8a, run8b, run4a, run4b, run1])

