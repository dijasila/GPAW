def agts(queue):
    run_special = queue.add('nwchem_95g2_1_pbe_def2_qzvppd_run_special.py',
                            queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                            ncpus=1, walltime=1*20, deps=[])
    run = queue.add('nwchem_95g2_1_pbe_def2_qzvppd_run.py',
                    queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                    ncpus=1, walltime=5*60, deps=[run_special])
    analyse = queue.add('nwchem_95g2_1_pbe_def2_qzvppd_analyse.py',
                        queueopts='-l nodes=1:ppn=1',
                        ncpus=1, walltime=5, deps=[run_special, run],
                        creates=['nwchem_95g2_1_pbe_def2_qzvppd_ae.csv',
                                 'nwchem_95g2_1_pbe_def2_qzvppd_energy.csv'])
