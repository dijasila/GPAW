config = {
    'scheduler': 'slurm',
    'extra_args': ['--account=project_462000135', '--mem=0'],
    'mpiexec': 'srun',
    'parallel_python': 'gpaw python',
    'nodes': [
        ('standard', {'cores': 256}),
        ('small', {'cores': 256}),
        ('debug', {'cores': 256}),
        ('largemem', {'cores': 256}),
        ('standard-g', {'cores': 128}),
        ('small-g', {'cores': 128}),
        ('dev-g', {'cores': 128})]}
