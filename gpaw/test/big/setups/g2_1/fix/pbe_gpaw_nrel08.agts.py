from gpaw.test.big.setups.g2_1.fix.pbe_gpaw_nrel08_analyse import tag
def agts(queue):
    run_generate = queue.add(tag + '_generate.py',
                             queueopts='-l nodes=1:ppn=4:ethernet',
                             ncpus=1,walltime=20, deps=[])
    run_special = [queue.add(tag + '_run_special.py %d' % i,
                             queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                             ncpus=1,
                             walltime=20*60,
                             deps=[run_generate])
                   for i in range(4)]
    run = [queue.add(tag + '_run.py %d' % i,
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,
                     walltime=40*60,
                     deps=run_special)
           for i in range(4)]
    analyse = queue.add(tag + '_analyse.py',
                        queueopts='-l nodes=1:ppn=1',
                        ncpus=1, walltime=5, deps=run,
                        creates=[tag + '_ea.csv',
                                 tag + '_energy.csv'])
    plot = queue.add(tag + '_plot.py',
                     queueopts='-l nodes=1:ppn=1',
                     ncpus=1, walltime=5, deps=[analyse],
                     creates=[tag + '.png'])
