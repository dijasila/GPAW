from myqueue.workflow import run


def workflow():
    run(script='adenine-thymine_complex_stack.py', cores=4, tmax='2h')
