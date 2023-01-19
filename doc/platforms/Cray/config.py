config = {
    'scheduler': 'slurm',
    'extra_args': ['--account=project_462000135', '--mem=0'],
    'mpiexec': 'srun',
    'parallel_python': 'gpaw python',
    'nodes': [
        ('standard', {'cores': 128}),
        ('small', {'cores': 128}),
        ('debug', {'cores': 128}),
        ('largemem', {'cores': 128}),
        ('standard-g', {'cores': 63}),
        ('small-g', {'cores': 63}),
        ('dev-g', {'cores': 63})]}
