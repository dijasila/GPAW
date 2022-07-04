class ResponseGroundStateAdapter:
    def __init__(self, calc):
        wfs = calc.wfs

        self.kd = wfs.kd
        self.world = calc.world
        self.gd = wfs.gd
        self.bd = wfs.bd
        self.pd = wfs.pd
        self.nspins = wfs.nspins
        self.dtype = wfs.dtype

        self.spos_ac = calc.spos_ac

        self.wfs = wfs
        self.kpt_u = wfs.kpt_u
        self.setups = wfs.setups
