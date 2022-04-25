from myqueue.workflow import run


def workflow():
    run(script='plot_freq.py')
    with run(script='silicon_ABS_simpleversion.py'):
        run(script='plot_silicon_ABS_simple.py')

    with run(script='silicon_ABS.py', cores=16, tmax='1h'):
        run(script='plot_ABS.py')

    with run(script='aluminum_EELS.py', cores=8, tmax='1h'):
        run(script='plot_aluminum_EELS_simple.py')

    with run(script='graphite_EELS.py', cores=8, tmax='1h'):
        run(script='plot_EELS.py')

    run(script='tas2_dielectric_function.py', cores=8, tmax='15m')
    run(script='graphene_dielectric_function.py', cores=8, tmax='15m')
