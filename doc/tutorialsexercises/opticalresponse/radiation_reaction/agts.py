def workflow():
    from myqueue.workflow import run
    td_jobs = [run(script='lcao_linres_NaD.py', tmax='1h'),
               run(script='lcao_linres_NaD2chain.py', tmax='1h')]

    # Plot spectra
    run(script='plot_spectra.py', deps=td_jobs)
    run(script='plot_dipoles.py', deps=td_jobs)
