from ase.utils.plugins import ExternalIOFormat
from ase import Atoms


gpaw_yaml = ExternalIOFormat(
    desc='GPAW-yaml output',
    code='+B',
    module='gpaw.yml',
    magic=b'#   __  _  _',
    glob=['*.yaml', '*.yml', '*.txt'])


def read_gpaw_yaml(fd, index):
    import yaml
    print(index)
    for dct in yaml.safe_load_all(fd):
        print(dct)
    return [Atoms('H')]
