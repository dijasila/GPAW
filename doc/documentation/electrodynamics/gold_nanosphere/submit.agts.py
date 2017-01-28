def agts(queue):
    c1 = queue.add('gold_nanosphere_calculate.py',
                   walltime=60)

    queue.add('plot.py',
              deps=c1,
              creates=['qsfdtd_vs_mie.png'])
