from myqueue.workflow import run


def workflow():
    run(script='pt_h2_tb_transport.py')
    with run(script='pt_h2_lcao_manual.py'):
        run(script='pt_h2_lcao_transport.py')
