def agts(queue):
    queue.add('plot_freq.py', creates='nl_freq_grid.png')
    queue.add('silicon_ABS_simpleversion.py', 
              creates=['df.csv', 'si_abs.png'])
    
    si = queue.add('silicon_ABS.py', creates=['si_abs.csv', 'mac_eps.csv'],
                   ncpus=16, walltime=100)
    queue.add('plot_ABS.py', deps=si, creates='silicon_ABS.png')
    
    al = queue.add('aluminum_EELS.py', creates=['eels.csv'],
                   ncpus=8, walltime=100)
    queue.add('plot_aluminum_EELS_simple.py', deps=al, creates=['aluminum_EELS.png'])
    
    GR = queue.add('graphite_EELS.py', 
                   creates=(['graphite_q_list'] + 
                            ['graphite_EELS_%d.csv' %i for i in range(1, 8)]),
                   ncpus=8, walltime=100)
    queue.add('plot_EELS.py', deps=GR, creates='graphite_EELS.png')
    
