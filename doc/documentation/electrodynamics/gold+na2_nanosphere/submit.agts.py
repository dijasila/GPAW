def agts(queue):
    c1 = queue.add('gold+na2_nanosphere_calculate.py',
                   ncpus=8,
                   walltime=60)

    c2 = queue.add('gold_nanosphere_calculate.py',
                   walltime=60)

    queue.add('plot_geom.py',
              deps=c1,
              creates=['geom.png'])

    queue.add('plot.py',
              deps=[c1, c2],
              creates=['qsfdtd_vs_mie.png', 'hybrid.png'])
