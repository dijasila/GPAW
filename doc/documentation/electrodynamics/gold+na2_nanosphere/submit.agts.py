def agts(queue):
    c1 = queue.add('calculate.py',
                   ncpus=8,
                   walltime=60)

    queue.add('plot_geom.py',
              deps=c1,
              creates=['geom.png'])

    queue.add('plot.py',
              deps=[c1],
              creates=['hybrid.png'])
