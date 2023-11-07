from myqueue.workflow import run


def workflow():
    with run(script='diamond_nv_minus.py', cores=16, tmax='4h'):
        run(script='json_to_csv.py')
    with run(script='biradical.py', cores=16, tmax='4h'):
        pass  # run(script='plot.py') OOM!!!
