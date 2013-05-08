from gpaw.test.big.g2_1.pbe_nwchem_def2_qzvppd_analyse import tag
def agts(queue):
    run = queue.add(tag + '_run.py',
                    queueopts='-l nodes=1:ppn=4:opteron4',
                    ncpus=1, walltime=6*60, deps=[])
    analyse = queue.add(tag + '_analyse.py',
                        ncpus=1, walltime=5, deps=[run],
                        creates=[tag + '_ea.csv',
                                 tag + '_energy.csv'])
    # optimization
    opt_run = queue.add(tag + '_opt_run.py',
                        queueopts='-l nodes=1:ppn=4:opteron4',
                        ncpus=1, walltime=22*60, deps=[])
    opt_analyse = queue.add(tag + '_opt_analyse.py',
                            ncpus=1, walltime=5, deps=[opt_run],
                            creates=[tag + '_opt_ea.csv',
                                     tag + '_opt_energy.csv',
                                     tag + '_opt_distance.csv'])
    opt_vs = queue.add(tag + '_opt_vs.py',
                       ncpus=1, walltime=5, deps=[opt_analyse],
                       creates=[tag + '_opt_ea_vs.csv',
                                tag + '_opt_distance_vs.csv'])
