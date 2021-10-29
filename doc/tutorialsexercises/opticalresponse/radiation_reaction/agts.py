def workflow():
    from myqueue.workflow import run
    run(script='lcao_linres_NaD.py')
    run(script='lcao_linres_NaD2chain.py')

    # Plot spectra
    run(script='plot_spectra.py')
    run(script='plot_dipoles.py')
