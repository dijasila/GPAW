from gpaw.lcao.eigensolver import DirectLCAO


class TBEigenSolver(DirectLCAO):
    def iterate(self, ham, wfs, occ=None) -> None:
        for kpt in wfs.kpt_u:
            self.iterate_one_k_point(ham, wfs, kpt, [wfs.Vt_qMM[kpt.q].copy()])
