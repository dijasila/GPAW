from gpaw.test.big.setups.g2_1.fix.pbe_nwchem_def2_qzvppd_analyse import tag
def agts(queue):
    run = queue.add(tag + '_run.py',
                    queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                    ncpus=1, walltime=6*60, deps=[])
    analyse = queue.add(tag + '_analyse.py',
                        queueopts='-l nodes=1:ppn=1',
                        ncpus=1, walltime=5, deps=[run],
                        creates=[tag + '_ea.csv',
                                 tag + '_energy.csv'])
