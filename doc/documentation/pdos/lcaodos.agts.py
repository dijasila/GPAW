def agts(queue):
    calc = queue.add('lcaodos_gs.py',
                     ncpus=8,
                     walltime=2)

    queue.add('lcaodos_gs.py',
              ncpus=1,
              walltime=5,
              deps=[calc])
