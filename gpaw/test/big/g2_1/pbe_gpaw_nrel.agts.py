from gpaw.test.big.g2_1.pbe_gpaw_nrel_analyse import tag
def agts(queue):
    run_generate = queue.add(tag + '_generate.py',
                             ncpus=1,walltime=20, deps=[])
    run_special = [queue.add(tag + '_run_special.py %d' % i,
                             queueopts='-l nodes=1:ppn=2:opteron:ethernet',
                             ncpus=1,
                             walltime=10*60,
                             deps=[run_generate])
                   for i in range(4)]
    run_cg = [queue.add(tag + '_run_cg.py %d' % i,
                             queueopts='-l nodes=1:ppn=2:opteron:ethernet',
                             ncpus=1,
                             walltime=10*60,
                             deps=[run_generate])
                   for i in range(4)]
    run = [queue.add(tag + '_run.py %d' % i,
                     queueopts='-l nodes=1:ppn=2:opteron:ethernet',
                     ncpus=1,
                     walltime=30*60,
                     deps=run_special + run_cg)
           for i in range(4)]
    if 1:  # run after releasing new setups
        analyse = queue.add(tag + '_analyse.py',
                            ncpus=1, walltime=5, deps=run,
                            creates=[tag + '_ea.csv',
                                     tag + '_energy.csv'])
        vs = queue.add(tag + '_vs.py',
                       ncpus=1, walltime=5, deps=[analyse],
                       creates=[tag + '_ea_vs.csv'])
        plot = queue.add(tag + '_plot.py',
                         ncpus=1, walltime=5, deps=[analyse],
                         creates=[tag + '_ea_vs.png'])
    # optimization
    if 0:  # run after releasing new setups
        opt_run = [queue.add(tag + '_opt_run.py %d' % i,
                             queueopts='-l nodes=1:ppn=2:opteron:ethernet',
                             ncpus=1,
                             walltime=40*60,
                             deps=[run_generate])
                   for i in range(20)]
    if 0:  # run after releasing new setups
        opt_analyse = queue.add(tag + '_opt_analyse.py',
                                ncpus=1, walltime=5, deps=opt_run,
                                creates=[tag + '_opt_ea.csv',
                                         tag + '_opt_energy.csv',
                                         tag + '_opt_distance.csv',
                                         tag + '_opt.csv'])
        opt_vs = queue.add(tag + '_opt_vs.py',
                           ncpus=1, walltime=5, deps=[opt_analyse],
                           creates=[tag + '_opt_ea_vs.csv',
                                    tag + '_opt_distance_vs.csv'])
