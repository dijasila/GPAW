optimizers = [
    'GoodOldQuasiNewtonD16',  # takes ~70h
    'BFGS',
    'LBFGS',
    'FIRE',
    #'MDMin',  # bad and memory grows to GBs!
    'BFGSLineSearch',
    'LBFGSLineSearch',
    #'SciPyFminCG',  # bad and memory grows to GBs!
    #'SciPyFminBFGS',  # bad
    'GoodOldQuasiNewton',
]

runsstr = ','.join(optimizers)

def agts(queue):
    run = [queue.add('g2_dzp.py %s' % o,
                     ncpus=1,
                     walltime=40*60)
           for o in optimizers*2]
    analyse = queue.add('task_analyse.py molecule g2_dzp ' + runsstr,
                        ncpus=1, walltime=10, deps=run,
                        creates=['g2_dzp_relaxed_energy.csv',
                                 'g2_dzp_optimizer_force_calls.png'])
