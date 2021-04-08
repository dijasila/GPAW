from myqueue.workflow import run


def workflow():
    spec_jobs = []
    with run(script='basis.py'):
        for dpath in ['', 'dzp']:
            with run(script='gs.py', cores=4, folder=dpath):
                td_jobs = []
                for kick in 'xyz':
                    td = run(script='td.py', args=['--kick', kick],
                             cores=4, tmax='1h', folder=dpath)
                    td_jobs.append(td)
                spec = run(script='spec.py', deps=td_jobs, folder=dpath)
                spec_jobs.append(spec)
    run(script='plot_spectra.py', deps=spec_jobs)
