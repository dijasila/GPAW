from gpaw.xc import XC


class XCFunctional:
    def __init__(self, params: dict, ncomponents: int):
        self.xc = XC(params, collinear=(ncomponents < 4))
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.no_forces = self.name.startswith('GLLB')
        self.type = self.xc.type

    def __str__(self):
        return f'name: {self.name}'

    def calculate(self, density, out) -> float:
        return self.xc.calculate(density.desc._gd, density.data, out.data)

    def calculate_paw_correction(self, setup, d, h):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self):
        return self.name
