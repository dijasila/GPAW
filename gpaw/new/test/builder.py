from gpaw.new.builder import DFTComponentsBuilder


class TestDFTComponentsBuilder(DFTComponentsBuilder):
    name = 'test'
    interpolation = ''

    def check_cell(self, cell):
        pass

    def create_basis_set(self):
        return None
        