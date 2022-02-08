from gpaw.xc import XC


class XCFunctional:
    def __init__(self, params: dict):
        self.xc = XC(params)
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.no_forces = self.name.startswith('GLLB')

    def calculate(self, density, out) -> float:
        return self.xc.calculate(density.desc._gd, density.data, out.data)

    def calculate_paw_correction(self, setup, d, h):
        return self.xc.calculate_paw_correction(setup, d, h)
