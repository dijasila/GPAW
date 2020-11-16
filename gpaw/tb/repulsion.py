class Repulsion:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, d):
        return (self.a + d * self.b) * np.exp(d / self.c)

    def fit(cls, d0, e, dedd, d2edd2):
        return Repulsion()
        