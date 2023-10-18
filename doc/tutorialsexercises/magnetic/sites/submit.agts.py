from myqueue.workflow import run


def workflow():
    run(script='Fe_site_properties.py', cores=40, tmax='1h')
