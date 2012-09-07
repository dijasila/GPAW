def agts(queue):
    run = [queue.add('pbe_jacapo.py %s' % r,
                     ncpus=4,
                     walltime=5*60)
           for r in range(1)]
    if 0:  # do not perform analysis
        # we keep once generated files static
        analyse = queue.add('analyse.py dcdft_pbe_jacapo',
                            ncpus=1, walltime=10, deps=run,
                            creates=['dcdft_pbe_jacapo.csv',
                                     'dcdft_pbe_jacapo.txt',
                                     'dcdft_pbe_jacapo_Delta.txt',
                                     'dcdft_pbe_jacapo_raw.csv'])
