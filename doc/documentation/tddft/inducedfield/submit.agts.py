from q2.job import Job


def workflow():
    return [
        Job('timepropagation_calculate.py@8x1m'),
        Job('timepropagation_continue.py@8x1m', deps=['timepropagation_calculate.py']),
        Job('timepropagation_postprocess.py@8x5s', deps=['timepropagation_continue.py']),
        Job('timepropagation_plot.py@1x5s', deps=['timepropagation_postprocess.py']),
        Job('casida_calculate.py@8x1m'),
        Job('casida_postprocess.py@8x5s', deps=['casida_calculate.py']),
        Job('casida_plot.py@1x5s', deps=['casida_postprocess.py'])]

def agts(queue):
    calc1 = queue.add('timepropagation_calculate.py',
                      ncpus=8,
                      walltime=60)

    calc2 = queue.add('timepropagation_continue.py',
                      ncpus=8,
                      walltime=60,
                      deps=calc1)

    calc3 = queue.add('timepropagation_postprocess.py',
                      ncpus=8,
                      walltime=5,
                      deps=calc2)
    
    calc4 = queue.add('timepropagation_plot.py',
                      ncpus=1,
                      walltime=5,
                      deps=calc3,  
                      creates=['na2_td_Ffe.png', 'na2_td_Frho.png', 'na2_td_Fphi.png'])

    calc5 = queue.add('casida_calculate.py',
                      ncpus=8,
                      walltime=60)

    calc6 = queue.add('casida_postprocess.py',
                      ncpus=8,
                      walltime=5,
                      deps=calc5)
    
    calc7 = queue.add('casida_plot.py',
                      ncpus=1,
                      walltime=5,
                      deps=calc6,
                      creates=['na2_casida_Ffe.png', 'na2_casida_Frho.png', 'na2_casida_Fphi.png'])
    
