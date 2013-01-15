if 0:
    from gpaw.test.big.dcdft.pbe_gpaw_pw import elements_slow
else:
    elements_slow = ['Mn']  # just test single element
def agts(queue):
    run16 = [queue.add('pbe_gpaw_pw.py %s' % s,
                       ncpus=16,
                       walltime=30*60)
             for s in elements_slow]
    run4 = [queue.add('pbe_gpaw_pw.py %s' % s,
                      ncpus=4,
                      walltime=40*60)
            for s in range(0)]  # use range(20)
    if 0:  # run when new setups ready
        analyse = queue.add('analyse.py dcdft_pbe_gpaw_pw',
                            ncpus=1, walltime=10, deps=run16 + run4,
                            creates=['dcdft_pbe_gpaw_pw.csv',
                                     'dcdft_pbe_gpaw_pw.txt',
                                     'dcdft_pbe_gpaw_pw_Delta.txt',
                                     'dcdft_pbe_gpaw_pw_raw.csv'])
        verify = queue.add('pbe_gpaw_pw_verify.py',
                           ncpus=1, walltime=10, deps=[analyse])
