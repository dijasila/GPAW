from gpaw.core import UniformGrid
from gpaw.new.modes import Mode


class LCAOMode(Mode):
    name = 'lcao'

    def create_wf_description(self,
                              grid: UniformGrid,
                              dtype) -> UniformGrid:
        raise NotImplementedError
