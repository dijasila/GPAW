def agts(queue):
    c1 = queue.add('calculate.py',
                   walltime=60)

    queue.add('plot.py',
              deps=c1,
              creates=['field.ind_Ffe.png'])
