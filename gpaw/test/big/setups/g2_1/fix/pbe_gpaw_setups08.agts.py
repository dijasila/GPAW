from gpaw.test.big.setups.g2_1.fix.pbe_gpaw_setups08_analyse import tag
def agts(queue):
    run_generate = queue.add(tag + '_run_generate.py',
                             queueopts='-l nodes=1:ppn=4:ethernet',
                             ncpus=1,walltime=20, deps=[])
    run_special = queue.add(tag + '_run_special.py',
                            queueopts='-l nodes=1:ppn=4:ethernet',
                            ncpus=1,walltime=10*60, deps=[run_generate])
    run1 = queue.add(tag +  '_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    run2 = queue.add(tag + '_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    run3 = queue.add(tag + '_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    run4 = queue.add(tag + '_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    analyse = queue.add(tag + '_analyse.py',
                        queueopts='-l nodes=1:ppn=1',
                        ncpus=1, walltime=5, deps=[run1, run2, run3, run4],
                        creates=[tag + '_ea.csv',
                                 tag + '_energy.csv'])
    plot = queue.add(tag + '_plot.py',
                     queueopts='-l nodes=1:ppn=1',
                     ncpus=1, walltime=5, deps=[analyse],
                     creates=[tag + '.png'])
