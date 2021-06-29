from gpaw import setup_paths
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter

# Insert the path to the created basis set
setup_paths.insert(0, '.')


def main(kick):
    kick_strength = [0., 0., 0.]
    kick_strength['xyz'.index(kick)] = 1e-5

    td_calc = LCAOTDDFT('gs.gpw', txt=f'td-{kick}.out')

    DipoleMomentWriter(td_calc, f'dm-{kick}.dat')

    # Origin: center of mass
    MagneticMomentWriter(td_calc, f'mm-COM-{kick}.dat',
                         origin='COM')

    # Origin: center of mass + 5 Å shift
    for shift_axis in 'xyz':
        origin_shift = [0, 0, 0]
        origin_shift['xyz'.index(shift_axis)] = 5
        MagneticMomentWriter(td_calc, f'mm-COM+{shift_axis}-{kick}.dat',
                             origin='COM', origin_shift=origin_shift)

    # Origin: arbitrary coordinate
    MagneticMomentWriter(td_calc, f'mm-123-{kick}.dat',
                         origin='zero', origin_shift=[1, 2, 3])

    td_calc.absorption_kick(kick_strength)
    td_calc.propagate(10, 2000)

    td_calc.write(f'td-{kick}.gpw', mode='all')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kick', default='x')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
