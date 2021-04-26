from myqueue.workflow import run


def workflow():
    with run(script='H2O_ir.py'):
        with run(script='H2O_rraman_calc.py', cores=4, tmax='1h'):
            run(script='H2O_rraman_spectrum.py')
            run(function=check)


def check():
    """Read result and make sure it's OK."""
    ...
