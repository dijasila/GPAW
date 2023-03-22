from myqueue.workflow import run


def workflow():
    with run(script='H.ralda_01_lda.py', cores=2, tmax='1m'):
        run(script='H.ralda_02_rpa_at_lda.py', cores=16, tmax='20m')
        run(script='H.ralda_03_ralda.py', cores=16, tmax='4h')
    with run(script='H.ralda_04_pbe.py', cores=2, tmax='1m'):
        run(script='H.ralda_05_rpa_at_pbe.py', cores=16, tmax='20m')
        run(script='H.ralda_06_rapbe.py', cores=16, tmax='4h')
    with run(script='CO.ralda_01_pbe_exx.py', cores=40, tmax='20m'):
        run(script='CO.ralda_02_CO_rapbe.py', cores=40, tmax='20h')
        run(script='CO.ralda_03_C_rapbe.py', cores=40, tmax='20h')
        run(script='CO.ralda_04_O_rapbe.py', cores=40, tmax='20h')
        with run(script='diamond.ralda_01_pbe.py', cores=16, tmax='1h'):
            run(script='diamond.ralda_02_rapbe_rpa.py', cores=40, tmax='6h')
    with run(script='diam_kern.ralda_01_lda.py', cores=8, tmax='1m'):
        with run(script='diam_kern.ralda_02_ralda_dens.py',
                 cores=8, tmax='1m'):
            with run(script='diam_kern.ralda_03_ralda_wave.py',
                     cores=8, tmax='1m'):
                with run(script='diam_kern.ralda_08_rpa.py',
                         cores=8, tmax='1m'):
                    run(script='diam_kern.ralda_09_compare.py',
                        cores=1, tmax='1m')
