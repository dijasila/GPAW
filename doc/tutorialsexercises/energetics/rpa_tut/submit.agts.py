from myqueue.workflow import run


def workflow():
    with run(script='gs_N2.py', cores=16):
        r1 = run(script='frequency.py', tmax='3h')
        r2 = run(script='con_freq.py', cores=2, tmax='16h')
        r3 = run(script='rpa_N2.py', cores=48, tmax='1h')
    with r1, r2:
        run(script='plot_w.py')
    with r2:
        run(script='plot_con_freq.py')
    with r3:
        run(script='extrapolate.py')

    with r1, r2, r3:
        run(shell='rm', args=['N.gpw', 'N2.gpw'])  # clean up

    with run(script='gs_graph_Co.py', cores=16):
        with run(script='rpa_graph_Co.py', cores=16, tmax='10h'):
            run(shell='rm', args=['gs_*.gpw'])  # clean up
