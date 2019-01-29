from gpaw import GPAW
from functools import partial
from gpaw.matrix import matrix_matrix_multiply as mmm
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab




class SternheimerResponse:


        def __init__(self, filename):
                self.restart_filename = filename
                self.calc = GPAW(filename, txt=None)
                #Check that all info is in file
                self.wfs = self.calc.wfs


                npts = self.wfs.kpt_u[0].psit.array.shape[1]
                #print("psit array shape", self.wfs.kpt_u[0].psit.array.shape)
                trial_potential = np.zeros((npts,npts))

                self.solve_sternheimer(trial_potential, 0)

                #test_wave = self.wfs.kpt_u[0].psit_nG[0]
                #self._apply_hamiltonian(test_wave, self.wfs.kpt_u[0])

        def chi0_response(self, num_eigs):
                start_vectors = self._get_trial_vectors(num_eigs)

                                



        def solve_sternheimer(self, potential, qvector):
                #print(len(self.wfs.kpt_u))
                #print(dir(self.wfs))
                #print(self.wfs.mykpts)
                #print(self.wfs.kpt_u)
                #print(dir(self.wfs.kpt_u[0]))
                #print(self.wfs.kpt_u[0].eps_n)
                #print((self.wfs.kpt_u[0].psit_nG).shape)
                #print(dir(self.wfs.kpt_u[0].psit))
                #print(self.wfs.kpt_u[0].psit.array.shape)
                #print(dir(self.wfs.kpt_u))
                #print("#############")
                #print(dir(self.wfs.kpt_u[0]))
                #print(self.wfs.kpt_u[0].__doc__)

                #print(dir(self.calc))
                #print(dir(self.wfs.kd))
                #print(self.wfs.kd.what_is.__doc__)
                #print(self.wfs.kd.where_is.__doc__)
                solution_pairs = []

                for index, kpt in enumerate(self.wfs.mykpts):
                        k_plus_q_vector = self.wfs.kd.bzk_kc[index] + qvector
                        

                        ##TODO find out how you are supposed to use this
                        k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [index])
                        assert len(k_plus_q_index) == 1
                        k_plus_q_index = k_plus_q_index[0]
                        k_plus_q_object = self.wfs.mykpts[k_plus_q_index]                       
                        potential_at_k = self._get_potential_at_kpt(potential, kpt)
                        for energy, psi in zip(kpt.eps_n, kpt.psit.array):
                                linop = self._get_LHS_linear_operator(k_plus_q_object, energy)
                                RHS = self._get_RHS(kpt, psi, potential_at_k)
                                

                                delta_wf, info = bicgstab(linop, RHS)

                                if info != 0:
                                        print(f"bicgstab did not converge. Info: {info}")


                                solution_pairs.append((delta_wf, psi))
                                #Solve Sternheimer for given n, q
                                #
                        #print("Looking at kpt")
                return solution_pairs

        def _get_potential_at_kpt(self, potential, kpt):

                ##shape = kpt.psit.array.shape[1] ##This seems buggy psit.array.shape is not the same as psit.myshape
                shape = kpt.psit.myshape[1]
                return np.zeros((shape,shape))


        def _kindex_to_kvector(self, kindex):
                coordinates = self.wfs.kd.bzk_kc[kindex]
                

        def _get_LHS_linear_operator(self, kpt, energy):
                def mv(v):
                        return self._apply_LHS_Sternheimer(v, kpt, energy)
                shape_tuple = (kpt.psit.array[0].shape[0], kpt.psit.array[0].shape[0])
                linop = LinearOperator(shape_tuple, matvec = mv)

                return linop


        def _apply_LHS_Sternheimer(self, wavefunction, kpt, energy):
                result = np.zeros_like(wavefunction)

                
                result = result + self._apply_valence_projector(wavefunction, kpt)
                
                result = result + self._apply_hamiltonian(wavefunction, kpt)
                
                result = result - (np.eye(result.shape[0])*energy).dot(wavefunction)

                return result

        def _apply_valence_projector(self, wavefunction, kpt):
                result = np.zeros_like(wavefunction)
                f_n = kpt.f_n/kpt.f_n.max()

                for index, wvf in enumerate(kpt.psit.array):
                        result = result + wvf*(wvf.conj().dot(wavefunction)*f_n[index])


                return result


        def _apply_hamiltonian(self, wavefunction, kpt):
                kpt.psit.read_from_file()
                H_partial = partial(self.wfs.apply_pseudo_hamiltonian, kpt, self.calc.hamiltonian)
                psit = kpt.psit
                #print(len(psit))
                #print("Len of work", len(self.wfs.work_array))
                tmp = psit.new(buf=self.wfs.work_array)
                H = self.wfs.work_matrix_nn
                P2 = kpt.P.new()

                psit.matrix_elements(operator=H_partial, result=tmp, out=H,
                                     symmetric=True, cc=True) ##!! Is cc=True correct here?
                self.calc.hamiltonian.dH(kpt.P, out = P2)
                
                a = mmm(1.0, kpt.P, "N", P2, "C", 1.0, H,  symmetric=True)
                #print("Type of a", type(a))
                self.calc.hamiltonian.xc.correct_hamiltonian_matrix(kpt, a.array)
                
                result = np.zeros_like(wavefunction)

                for index1, wvf1 in enumerate(psit.array):
                        for index2, wvf2, in enumerate(psit.array):
                                inner_prod = wvf2.conj().dot(wavefunction)
                                matrix_element = a.array[index1, index2]
                                result = result + wvf1*matrix_element*inner_prod

                return result
                
        def _get_RHS(self, kpt, psi, potential):
                RHS = np.zeros_like(kpt.psit.array[0])
                RHS = -(potential.dot(psi) - self._apply_valence_projector(potential.dot(psi), kpt))

                return RHS
                





if __name__=="__main__":
        filen = "test.gpw"
        import os
        if os.path.isfile(filen):
                print("Reading file")
                respObj = SternheimerResponse(filen)
        else:
                print("Generating new test file")
                from ase import Atoms
                from ase.build import bulk
                from gpaw import PW, FermiDirac
                a = 10
                c = 5
                d = 0.74
                #atoms = Atoms("H2", positions=([c-d/2, c, c], [c+d/2, c,c]),
                #       cell = (a,a,a))
                atoms = bulk("Si", "diamond", 5.43)
                calc = GPAW(mode=PW(100),
                            xc ="PBE",
                            kpts=(8,8,8),
                            random=True,                            
                            occupations=FermiDirac(0.01),
                            txt = "test.out")
                
                atoms.set_calculator(calc)
                energy = atoms.get_potential_energy()
                calc.write(filen, mode = "all")
 
                respObj = SternheimerResponse(filen)
                
                
