from gpaw.tddft import TDDFT, DipoleMomentWriter, MagneticMomentWriter


def main(kick):
    kick_strength = [0., 0., 0.]
    kick_strength['xyz'.index(kick)] = 1e-5

    td_calc = TDDFT('gs.gpw',
                    solver=dict(name='CSCG', tolerance=1e-8),
                    txt=f'td-{kick}.out')

    DipoleMomentWriter(td_calc, f'dm-{kick}.dat')
    MagneticMomentWriter(td_calc, f'mm-{kick}.dat')

    td_calc.absorption_kick(kick_strength)
    td_calc.propagate(10, 2000)

    td_calc.write(f'td-{kick}.gpw', mode='all')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kick', default='x')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
