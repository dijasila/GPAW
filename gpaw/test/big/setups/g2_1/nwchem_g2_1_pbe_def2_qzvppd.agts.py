def agts(queue):
    run = queue.add('nwchem_g2_1_pbe_def2_qzvppd_run.py',
                    queueopts='-l nodes=1:ppn=4:opteron:ethernet',
                    ncpus=1, walltime=5*60, deps=[])
    analyse = queue.add('nwchem_g2_1_pbe_def2_qzvppd_analyse.py',
                        queueopts='-l nodes=1:ppn=1',
                        ncpus=1, walltime=5, deps=[run],
                        creates=['nwchem_g2_1_pbe_def2_qzvppd_ae.csv',
                                 'nwchem_g2_1_pbe_def2_qzvppd_energy.csv'])
