from gpaw.density import Density

class GLLBDensity(Density):
    def __init__(self, paw, magmom_a, gllb_xc):
        """Create the GLLBDensity object."""

        Density.__init__(self, paw, magmom_a)
        self.gllb_xc = gllb_xc

    def update(self, kpt_u, symmetry):
        """In addition to Density.update:
           Calculate the response matrices to be used in GLLB-calculations. """

        Density.update(self, kpt_u, symmetry)

        # Compute atomic "response-density" matrices:
        for nucleus in self.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            Dresp_sii = zeros((self.nspins, ni, ni), Float)
            for kpt in kpt_u:
                P_ni = nucleus.P_uni[kpt.u]
                w = self.gllb_xc.get_weights_kpoint(kpt)
                Dresp_sii[kpt.s] += real(dot(cc(transpose(P_ni)),
                                             P_ni * kpt.f_n[:, NewAxis] * w[:, NewAxis]))
            nucleus.Dresp_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.kpt_comm.sum(nucleus.Dresp_sp)

        comm = self.gd.comm
        
        if symmetry is not None:
            Dresp_asp = []
            for nucleus in self.nuclei:
                if comm.rank == nucleus.rank:
                    Dresp_sp = nucleus.Dresp_sp
                    comm.broadcast(Dresp_sp, nucleus.rank)
                else:
                    ni = nucleus.get_number_of_partial_waves()
                    np = ni * (ni + 1) / 2
                    D_sp = zeros((self.nspins, np), Float)
                    comm.broadcast(Dresp_sp, nucleus.rank)
                Dresp_asp.append(Dresp_sp)

            for s in range(self.nspins):
                Dresp_aii = [unpack2(Dresp_sp[s]) for Dresp_sp in Dresp_asp]
                for nucleus in self.my_nuclei:
                    nucleus.symmetrize(Dresp_aii, symmetry.maps, s, response = True)

