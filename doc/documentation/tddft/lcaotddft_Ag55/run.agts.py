from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=48, tmax='30m'):
        run(script='td.py', cores=48, tmax='4h'):
            spec = run(script='spec.py', deps=[td]):
                plot = run(script='plot_spec.py', deps=[spec])
    basis_my = run(script='basis.py', folder='mybasis')
    gs_my = run(script='gs.py', cores=48, tmax='30m', folder='mybasis', deps=[basis_my])
    td_my = run(script='td.py', cores=48, tmax='4h', folder='mybasis', deps=[gs_my])
    spec_my = run(script='spec.py', folder='mybasis', deps=[td_my])
    gs_dzp = run(script='gs.py', cores=48, tmax='2h', folder='dzp')
    td_dzp = run(script='td.py', cores=48, tmax='4h', folder='dzp', deps=[gs_dzp])
    spec_dzp = run(script='spec.py', folder='dzp', deps=[td_dzp])
    basis_plot = run(script='plot_spec_basis.py'
                      deps=[spec, spec_my, spec_dzp])

    return [gs, td, spec, plot,
            basis_my, gs_my, td_my, spec_my,
            gs_dzp, td_dzp, spec_dzp,
            basis_plot]
