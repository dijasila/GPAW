from gpaw.new.brillouin import IBZ


class Symmetry:
    def __init__(self, symmetry):
        self.symmetry = symmetry

    def reduce(self, bz):
        return IBZ(self, bz, [0], [0], [1.0])

    def __eq__(self, other):
        s1 = self.symmetry
        s2 = other.symmetry
        return (len(s1.op_scc) == len(s2.op_scc) and
                (s1.op_scc == s2.op_scc).all() and
                (s1.ft_sc == s2.ft_sc).all() and
                (s1.a_sa == s2.a_sa).all())
