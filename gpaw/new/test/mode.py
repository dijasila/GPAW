from gpaw.new.modes import Mode


class TestMode(Mode):
    name = 'test'
    interpolation = ''

    def check_cell(self, cell):
        pass
