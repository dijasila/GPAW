class XCFunctional:
    def __init__(self, xc):
        self.xc = xc
        self.setup_name = xc.get_setup_name()

    def calculate(self, density, out) -> float:
        return self.xc.calculate(density.grid._gd, density.data, out.data)

    def calculate_paw_correction(self, setup, d, h):
        return self.xc.calculate_paw_correction(setup, d, h)
