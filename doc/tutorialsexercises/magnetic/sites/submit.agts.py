from myqueue.workflow import run


def workflow():
    with run(script='Fe_site_properties.py', cores=40, tmax='1h'):
        run(script='Fe_plot_site_properties.py')
