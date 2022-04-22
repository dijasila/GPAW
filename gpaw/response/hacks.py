class GaGb:
    def __init__(self, blockcomm, nG):
        self.blockcomm = blockcomm
        mynG = (nG + blockcomm.size - 1) // blockcomm.size
        self.Ga = min(blockcomm.rank * mynG, nG)
        self.Gb = min(self.Ga + mynG, nG)
        self.nGlocal = self.Gb - self.Ga
        self.nG = nG

        self.myslice = slice(self.Ga, self.Gb)
