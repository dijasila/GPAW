def agts(queue):
    c1 = queue.add('calculate.py',
                   ncpus=8,
                   walltime=60)

    c2 = queue.add('postprocess.py',
                   ncpus=8,
                   walltime=10,
                   deps=c1)

    queue.add('plot.py',
              deps=c2,
              creates=['cl_field.ind_Ffe.png', 'qm_field.ind_Ffe.png',
                       'tot_field.ind_Ffe.png'])
