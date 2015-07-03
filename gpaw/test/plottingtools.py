import numpy as np

from gpaw.response.df import DielectricFunction
from gpaw.response.integrators import TetrahedronIntegrator 
from gpaw.response.chi0 import ArrayDescriptor

omega_w = np.array([0.4, 0.5], float) # / Hartree
wd = ArrayDescriptor(omega_w)

calc = 'gs.gpw'
df = DielectricFunction(calc)
chi0 = df.chi0

pd, PWSA = chi0.get_PWDescriptor([0, 0, 0])
bzk_kv = chi0.get_kpoints(pd)

# Initialize integrator
integrator = TetrahedronIntegrator()
td = integrator.tesselate(bzk_kv)

# Integrate interband response
m1 = chi0.nocc1
m2 = chi0.nbands
kd = chi0.calc.wfs.kd
n1 = 0
n2 = chi0.nocc2
eig_kwargs = {'kd': kd, 'm1': m1, 'm2': m2, 'n1': 0,
              'n2': n2, 'pd': pd}

get_eigenvalues = chi0.get_eigenvalues

# Relevant quantities
bzk_kc = td.points
nk = len(bzk_kc)

# Store eigenvalues
deps_Mk = None
for K in range(nk):
    k_c = bzk_kc[K]
    deps_M = get_eigenvalues(k_c)
    if deps_Mk is None:
        deps_Mk = np.zeros((len(deps_M), nk), float)
    deps_Mk[:, K] = deps_M

# Store indices for frequencies
indices_SMi = np.zeros((td.nsimplex, deps_M.shape[0], 2), int)

for s, simplex in enumerate(td.simplices):
    teteps_Mk = deps_Mk[:, simplex]
    emin_M, emax_M = teteps_Mk.min(1), teteps_Mk.max(1)
    i0_M, i1_M = wd.get_index_range(emin_M, emax_M)
    indices_SMi[s, :, 0] = i0_M
    indices_SMi[s, :, 1] = i1_M

omega_w = wd.get_data()
nw = len(wd)

patches_wP = [[] for w in range(nw)]

for S, indices_Mi in enumerate(indices_SMi):
    teteps_Mk = deps_Mk[:, td.simplices[S]]
    tetk_kc = td.points[td.simplices[S]]
    for M, indices_i in enumerate(indices_Mi):
        teteps_k = teteps_Mk[M]
        for iw in range(indices_i[0], indices_i[1]):
            o = omega_w[iw]
            patch = []
            for k1 in range(4):
                for k2 in range(k1, 4):
                    if teteps_k[k1] < o and teteps_k[k2] > o:
                        a = ((o - teteps_k[k1]) /
                             (teteps_k[k2] - teteps_k[k1]))
                        k_c = tetk_kc[k1] * (1 - a) + tetk_kc[k2] * a
                        patch.append(k_c)
                    elif teteps_k[k1] > o and teteps_k[k2] < o:
                        a = ((o - teteps_k[k2]) /
                             (teteps_k[k1] - teteps_k[k2]))
                        k_c = tetk_kc[k2] * (1 - a) + tetk_kc[k1] * a
                        patch.append(k_c)
            patches_wP[iw].append(np.array(patch))

print(patches_wP)
