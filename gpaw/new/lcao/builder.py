from gpaw.core import UniformGrid


class LCAOMode:
    name = 'lcao'

    def create_wf_description(self,
                              grid: UniformGrid,
                              dtype) -> UniformGrid:
        raise NotImplementedError
