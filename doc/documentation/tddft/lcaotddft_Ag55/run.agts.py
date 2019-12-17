from myqueue.task import task


def create_tasks():
    gs = task('gs.py@48:30m'),
    td = task('td.py@48:4h', deps=[gs]),
    spec = task('spec.py@1:1m', deps=[td]),
    plot = task('plot_spec.py@1:1m', deps=[spec]),
    basis_my = task('basis.py@1:10m', folder='mybasis'),
    gs_my = task('gs.py@48:30m', folder='mybasis', deps=[basis_my]),
    td_my = task('td.py@48:4h', folder='mybasis', deps=[gs_my]),
    spec_my = task('spec.py@1:1m', folder='mybasis', deps=[td_my]),
    gs_dzp = task('gs.py@48:30m', folder='dzp'),
    td_dzp = task('td.py@48:4h', folder='dzp', deps=[gs_dzp]),
    spec_dzp = task('spec.py@1:1m', folder='dzp', deps=[td_dzp]),
    basis_plot = task('plot_spec_basis.py@1:1m',
                      deps=[spec, spec_my, spec_dzp]),

    return [gs, td, spec, plot,
            basis_my, gs_my, td_my, spec_my,
            gs_dzp, td_dzp, spec_dzp,
            basis_plot]
