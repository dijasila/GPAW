from pathlib import Path

from myqueue.workflow import run


def workflow():
    with run(script='gs_BaTiO3.py', cores=8, tmax='30m'):
        with run(script='polarization_BaTiO3.py', cores=8, tmax='1h'):
            run(function=check)
        with run(script='born_BaTiO3.py', cores=8, tmax='10h'):
            run(script='get_borncharges.py')

    with run(script='gs_Sn.py', cores=8, tmax='30m'):
        with run(script='Sn_parallel_transport.py', cores=8, tmax='5h'):
            run(script='plot_phase.py')


def check():
    txt = Path('polarization_BaTiO3.out').read_text()
    # will fix this after next AGTS-run!
    assert txt == ...
