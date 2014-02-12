class NeedsGD:
    def __init__(self):
        self.gd = None

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def allocate(self):
        pass

    def estimate_memory(self, mem):
        raise NotImplementedError
