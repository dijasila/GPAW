def agts(queue):
    c1 = queue.add('gold_nanosphere_inducedfield.py',
                   walltime=60)

    queue.add('gold_nanosphere_inducedfield_plot.py',
              deps=c1,
              creates=['field.ind_Ffe.png'])
