import numpy as np
from collections import defaultdict


class ClebschGordanCalculator:
    def __init__(self):
        self.coefficients = defaultdict(float)
        self._calculated = set()

    def calculate(self, j1,m1,j2,m2,j,m):
        if (j1, j2, j) in self._calculated:
            return self.coefficients[(j1,m1,j2,m2,j,m)]
        else:
            self._subroutine(j1, j2, j)
            self._calculated.add((j1,j2,j))
            return self.coefficients[(j1,m1,j2,m2,j,m)]
            
            
    def combo_possibilites(self, j1, j2):
        return list(range(np.abs(j1-j2), j1+j2+1))


    def _subroutine(self, j1, j2, j):
        '''
        Calculate all non-zero Clebsch-Gordan coefficients j1,j2
        '''
        self.coefficients[(j1,j1,j2,j-j1,j,j)] = 1



        for ind1, m1 in enumerate(np.arange(j1, -j1 - 1, -1)):
            for ind2, m2 in enumerate(np.arange(j2, -j2 - 1, -1)):
                if np.abs(m1 + m2) > j:
                    self.coefficients[(j1, m1, j2, m2, j, m1+m2)] = 0
                    continue
                if m1 == j1:
                    if m2 > j - j1 -1:
                        continue
                    self.coefficients[(j1, m1, j2, m2, j, m1+m2)] = self.minusrelation(j1, m1, j2, m2, j, m1+m2)
                else:				
                    self.coefficients[(j1, m1, j2, m2, j, m1+m2)] = self.plusrelation(j1, m1, j2, m2, j, m1+m2)



        #Normalize the coefficients
        for m in np.arange(-j, j+1):
            norm = 0
            for m1 in np.arange(-j1, j1+1):
                for m2 in np.arange(-j2, j2+1):
                    if (j1, m1, j2, m2, j, m) in self.coefficients:
                        norm += np.abs(self.coefficients[(j1,m1,j2,m2,j,m1+m2)])**2
            for m1 in np.arange(-j1, j1+1):
                for m2 in np.arange(-j2, j2+1):
                    if (j1, m1, j2, m2, j, m) in self.coefficients:
                        self.coefficients[(j1,m1,j2,m2,j,m)] = self.coefficients[(j1,m1,j2,m2,j,m)]/np.sqrt(norm)



    def minusrelation(self, j1,m1,j2,m2,j,m):
        '''
        This is from the recursion relation that can be derived by applying J_- (the lowering operator) to both sides of |j1j2,JM> = |j1m1,j2m2><j1m1,j2m2|j1j2,JM>
        '''
        if (m1 +1) <= j1 and (m2 + 1 + m1) <= j and (m2 + 1) <= j2:
            return (np.sqrt((j1+m1+1)*(j1-m1))*self.coefficients[(j1,m1+1, j2, m2, j, m+1)] + np.sqrt((j2+m2+1)*(j2-m2))*self.coefficients[(j1,m1,j2,m2+1, j,m+1)])/np.sqrt((j+m+1)*(j-m))
        elif (m2+1+m1) <= j and (m2+1) <= j2:
            return (np.sqrt((j2+m2+1)*(j2-m2))*self.coefficients[(j1,m1,j2,m2+1, j,m+1)])/np.sqrt((j+m+1)*(j-m))
        elif (m1 + 1) <= j1 and (m1 + m2 + 1) <= j:
            return (np.sqrt((j1+m1+1)*(j1-m1))*self.coefficients[(j1,m1+1, j2, m2, j, m+1)])/np.sqrt((j+m+1)*(j-m))
        else:
            return 0

    def plusrelation(self, j1, m1, j2, m2, j, m):
        '''
        This is from the recursion relation that can be derived by applying J_+ (the raising operator) to both sides of |j1j2,JM> = |j1m1,j2m2><j1m1,j2m2|j1j2,JM>
        '''
        if (m1 + 1) <= j1 and (m2 -1) >= -j2 and (m1 + m2 + 1) <= j and (m1 + 1 + m2 - 1) >= -j:
            return (np.sqrt((j-m)*(j+m+1))*self.coefficients[(j1, m1+1,j2,m2,j,m+1)] - np.sqrt((j2-m2+1)*(j2+m2))*self.coefficients[(j1,m1+1,j2,m2-1, j, m)])/(np.sqrt((j1-m1)*(j1+m1+1)))
        elif (m1 + 1) <= j1 and (m1 + m2 + 1) <= j:
            return (np.sqrt((j-m)*(j+m+1))*self.coefficients[(j1, m1+1,j2,m2,j,m+1)])/(np.sqrt((j1-m1)*(j1+m1+1)))
        elif (m2 - 1) >= -j2 and (m1 + m2) >= -j:
            return -(np.sqrt((j2-m2+1)*(j2+m2))*self.coefficients[(j1,m1+1,j2,m2-1, j, m)])/(np.sqrt((j1-m1)*(j1+m1+1)))
        else:
            return 0


    def _test(self, start_j, end_j):
        '''
        Test against sympys implementation of clebsch-gordan coefficients
        The reason we do not just use sympys implementation is that it would add another dependency to GPAW.
        '''
        import sympy.physics.quantum.cg as cg
        def exact(j1, m1, j2, m2, j, m):
            return float(cg.CG(j1,m1,j2,m2,j,m).doit())
        import time
        t1 = time.time()
        for j1 in range(start_j, end_j):
            for j2 in range(start_j, end_j):
                for j in range(np.abs(j1-j2), j1+j2+1):
                    for m1 in range(-j1, j1+1):
                        for m2 in range(-j2, j2+1):
                            for m in range(-j, j+1):						
                                assert np.allclose(self.calculate(j1,m1,j2,m2,j,m), exact(j1,m1,j2,m2,j,m), atol = 1e-5)

        t2 = time.time()
        print("Testing all possible combinations from angular momentum {} to {} took {} seconds.".format(start_j, end_j, t2-t1))
