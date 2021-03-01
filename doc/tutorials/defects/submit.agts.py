from myqueue.workflow import run


def workflow():
    runs = [run(script='gaas.py', args=[1], tmax='1h'),
            run(script='gaas.py', args=[2], cores=8, tmax='1h'),
            run(script='gaas.py', args=[3], cores=24, tmax='2h'),
            run(script='gaas.py', args=[4], cores=48, tmax='24h')]
    with run(script='electrostatics.py', cores=8, processes=1, tmax='15m',
             deps=runs):
        run(script='plot_energies.py')
        run(script='plot_potentials.py')

    runs = [run(script='BN.py', args=[1], cores=8, tmax='1h'),
            run(script='BN.py', args=[2], cores=8, tmax='1h'),
            run(script='BN.py', args=[3], cores=8, tmax='2h'),
            run(script='BN.py', args=[4], cores=24, tmax='2h'),
            run(script='BN.py', args=[5], cores=48, tmax='2h')]
    with run(script='electrostatics_BN.py', tmax='15m', deps=runs):
        run(script='plot_energies_BN.py')
        run(script='plot_potentials_BN.py')
        run(script='plot_epsilon.py')
