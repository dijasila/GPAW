from myqueue.workflow import run


def workflow():
    with run(script='h2_gs.py'):
        run(script='h2_diss.py', cores=8)
    with run(script='graphene_h_gs.py', cores=8):
        run(script='graphene_h_prop.py', cores=32, tmax='2h')
