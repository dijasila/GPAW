def workflow():
    from myqueue.workflow import run
    run(script='con_pw.py')
    run(script='stress.py')
