from gpaw.utilities.grid_redistribute import rigorous_testing

failures = rigorous_testing()
assert len(failures) == 0
