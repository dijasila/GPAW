from ase.utils.plugins import ExternalIOFormat

gpaw_yaml = ExternalIOFormat(
    desc='GPAW-yaml output',
    code='+B',
    module='gpaw.yml',
    magic=b'#  __  _  _')
