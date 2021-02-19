if 1:  # example: set eigensolver/mixer
    from gpaw.eigensolvers.davidson import Davidson
    from gpaw import MixerSum

    def calc(atoms):
        atoms.calc.set(eigensolver=Davidson(), mixer=MixerSum(0.05, 5, 50))
