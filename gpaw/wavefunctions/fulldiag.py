from __future__ import print_function, division


def fulldiag(filename, nbands=None, scalapack=1, dryrun=False):
    from gpaw import GPAW
    calc = GPAW(filename,
                parallel={'band': scalapack},
                txt=filename[:-3] + 'full.txt')
    if not dryrun:
        calc.diagonalize_full_hamiltonian(nbands)
        calc.write(filename[:-3] + 'full.gpw', 'all')
    
    return calc.wfs.pd.ngmax

description = """\
Set up full H and S matrices and find all or some eigenvectors/values."""


def main():
    import optparse
    parser = optparse.OptionParser(usage='Usage: %prog <gpw-file> [options]',
                                   description=description)
    add = parser.add_option
    
    add('-n', '--bands', type=int,
        help='Number of bands to calculate.  Defaults to all.')
    add('-s', '--scalapack', type=int, default=1,
        help='Number of cores to use for ScaLapack.  Default is one.')
    add('-d', '--dry-run', action='store_true')
    
    opts, args = parser.parse_args()
    if len(args) != 1:
        parser.error('No gpw-file!')
    assert args[0].endswith('.gpw')
    
    ng = fulldiag(args[0], opts.bands, opts.scalapack, opts.dry_run)
    mem = ng**2 * 16 / 1024**2
    print('Maximum matrix size: {0}x{0}={1:.3f} MB'.format(ng, mem))


if __name__ == '__main__':
    main()
