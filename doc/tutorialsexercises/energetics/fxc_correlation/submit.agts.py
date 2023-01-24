from myqueue.workflow import run

def workflow():
    with run(script='H.ralda_01_lda.py', cores=2, tmax='5m'):
        run(script='H.ralda_02_rpa_at_lda.py', cores=16, tmax='20m')
        run(script='H.ralda_03_ralda.py', cores=16, tmax='200m')
    with run(script='H.ralda_04_pbe.py', cores=2, tmax='5m'):
        run(script='H.ralda_05_rpa_at_pbe.py', cores=16, tmax='20m')
        run(script='H.ralda_06_rapbe.py', cores=16, tmax='200m')
    with run(script='CO.ralda_01_pbe_exx.py', cores=16, tmax='1000m'):
        run(script='CO.ralda_02_CO_rapbe.py', cores=16, tmax='2000m')
        run(script='CO.ralda_03_C_rapbe.py', cores=16, tmax='2000m')
        run(script='CO.ralda_04_O_rapbe.py', cores=16, tmax='2000m')
    with run(script='diamond.ralda_01_pbe.py', cores=1, tmax='100m'):
        run(script='diamond.ralda_02_rapbe_rpa.py', cores=16, tmax='1200m')
    with run(script='diam_kern.ralda_01_lda.py', cores=8, tmax='2m'):
        with run(script='diam_kern.ralda_03_ralda_wave.py', cores=8, tmax='5m'):
            with run(script='diam_kern.ralda_08_rpa.py', cores=8, tmax='5m'):
                run(script='diam_kern.ralda_09_compare.py', cores=1, tmax='5m')
