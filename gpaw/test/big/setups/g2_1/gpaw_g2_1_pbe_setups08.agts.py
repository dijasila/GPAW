def agts(queue):
    run_special = queue.add('gpaw_g2_1_pbe_setups08_run_special.py',
                            queueopts='-l nodes=1:ppn=1:ethernet',
                            ncpus=1,walltime=6*60, deps=[])
    run1 = queue.add('gpaw_g2_1_pbe_setups08_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    run2 = queue.add('gpaw_g2_1_pbe_setups08_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    run3 = queue.add('gpaw_g2_1_pbe_setups08_run.py',
                     queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                     ncpus=1,walltime=20*60, deps=[run_special])
    analyse = queue.add('gpaw_g2_1_pbe_setups08_analyse.py',
                        queueopts='-l nodes=1:ppn=1',
                        ncpus=1, walltime=5, deps=[run_special, run1, run2, run3],
                        creates=['gpaw_g2_1_pbe_setups08_ae.csv',
                                 'gpaw_g2_1_pbe_setups08_energy.csv'])
