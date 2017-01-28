def agts(queue):
    c1 = queue.add('gold+na2_nanosphere_inducedfield.py',
                   ncpus=8,
                   walltime=60)

    c2 = queue.add('inducedfield_postprocess.py',
                   ncpus=8,
                   walltime=10,
                   deps=c1)

    queue.add('inducedfield_plot.py',
              deps=c2,
              creates=['cl_field.ind_Ffe.png', 'qm_field.ind_Ffe.png',
                       'tot_field.ind_Ffe.png'])
