if 0:  # example: set eigensolver/mixer
    from gpaw.eigensolvers.davidson import Davidson
    from gpaw import Mixer, MixerSum
    def calc(atoms):
        if atoms.get_initial_magnetic_moments().any():
            M = MixerSum
        else:
            M = Mixer
        atoms.calc.set(eigensolver=Davidson(niter=1, normalize=True),
                       mixer=M(0.05, 5, 50))
