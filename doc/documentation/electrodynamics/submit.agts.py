def agts(queue):
    calc1 = queue.add('gold+na2_nanosphere_calculate.py',
                      ncpus=8,
                      walltime=60)

    queue.add('plot_geom.py',
              ncpus=1,
              walltime=5,
              deps=calc1,
              creates=['geom.png'])

    calc3 = queue.add('gold_nanosphere_calculate.py',
                      ncpus=1,
                      walltime=60)

    queue.add('plot.py',
              ncpus=1,
              walltime=5,
              deps=[calc1, calc3],
              creates=['qsfdtd_vs_mie.png', 'hybrid.png'])

    clind1 = queue.add('gold_nanosphere_inducedfield.py',
                       ncpus=1,
                       walltime=60)

    queue.add('gold_nanosphere_inducedfield_plot.py',
              ncpus=1,
              walltime=10,
              deps=clind1,
              creates=['field.ind_Ffe.png'])

    ind1 = queue.add('gold+na2_nanosphere_inducedfield.py',
                     ncpus=8,
                     walltime=60)

    ind2 = queue.add('inducedfield_postprocess.py',
                     ncpus=8,
                     walltime=10,
                     deps=ind1)

    queue.add('inducedfield_plot.py',
              ncpus=1,
              walltime=5,
              deps=ind2,
              creates=['cl_field.ind_Ffe.png', 'qm_field.ind_Ffe.png', 'tot_field.ind_Ffe.png'])
    
    queue.add('plot_permittivity.py',
              ncpus=1,
              walltime=5,
              creates=['Au.yml.png'])
