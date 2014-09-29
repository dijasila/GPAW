def agts(queue):
    calc1 = queue.add('gold+na2_nanosphere_calculate.py',
                     ncpus=8,
                     walltime=60)

    calc2 = queue.add('plot_geom.py',
                      ncpus=1,
                      walltime=5,
                      deps=calc1,
                      creates=['geom.png'])

    calc3 = queue.add('gold_nanosphere_calculate.py',
                      ncpus=8,
                      walltime=60)

    calc4 = queue.add('plot.py',
                      ncpus=1,
                      walltime=5,
                      deps=calc3,
                      creates=['qsfdtd_vs_mie.png', 'hybrid.png'])

