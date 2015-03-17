# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
from __future__ import print_function
import numpy as np
from math import pi

def run():
    from gpaw.spherical_harmonics import Y
    weight_n = np.zeros(50)
    Y_nL = np.zeros((50, 25))
    R_nv = np.zeros((50, 3))

    # We use 50 Lebedev quadrature points which will integrate
    # spherical harmonics correctly for l < 12:
    n = 0
    for v in [0, 1, 2]:
        for s in [-1, 1]:
            R_nv[n, v] = s
            n += 1
    C = 2**-0.5
    for v1 in [0, 1, 2]:
        v2 = (v1 + 1) % 3
        v3 = (v1 + 2) % 3
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                R_nv[n, v2] = s1 * C
                R_nv[n, v3] = s2 * C
                n += 1
    C = 3**-0.5
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                R_nv[n] = (s1 * C, s2 * C, s3 * C) 
                n += 1
    C1 = 0.30151134457776357
    C2 = (1.0 - 2.0 * C1**2)**0.5
    for v1 in [0, 1, 2]:
        v2 = (v1 + 1) % 3
        v3 = (v1 + 2) % 3
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    R_nv[n, v1] = s1 * C2
                    R_nv[n, v2] = s2 * C1
                    R_nv[n, v3] = s3 * C1
                    n += 1
    assert n == 50

    weight_n[0:6] = 0.01269841269841270
    weight_n[6:18] = 0.02257495590828924
    weight_n[18:26] = 0.02109375000000000
    weight_n[26:50] = 0.02017333553791887
                  
    n = 0
    for x, y, z in R_nv:
        Y_nL[n] = [Y(L, x, y, z) for L in range(25)]
        n += 1

    # Write all 50 points to an xyz file as 50 hydrogen atoms:
    f = open('50.xyz', 'w')
    f.write('50\n\n')
    for x, y, z in R_nv:
        print('H', 4*x, 4*y, 4*z, file=f)

    #print np.dot(weights, Y_nL) * (4*pi)**.5

    print('weight_n = np.array(%s)' % weight_n.tolist())
    print('Y_nL = np.array(%s)' % Y_nL.tolist())
    print('R_nv = np.array(%s)' % R_nv.tolist())

    return weight_n, Y_nL, R_nv

if __name__ == '__main__':
    run()

weight_n = np.array([0.0126984126984127, 0.0126984126984127, 0.0126984126984127, 0.0126984126984127, 0.0126984126984127, 0.0126984126984127, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.02257495590828924, 0.021093750000000001, 0.021093750000000001, 0.021093750000000001, 0.021093750000000001, 0.021093750000000001, 0.021093750000000001, 0.021093750000000001, 0.021093750000000001, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887, 0.02017333553791887])
Y_nL = np.array([[0.28209479177387814, 0.0, 0.0, -0.48860251190291992, 0.0, 0.0, -0.31539156525252005, 0.0, 0.54627421529603959, 0.0, 0.0, 0.0, 0.0, 0.45704579946446577, 0.0, -0.59004358992664352, 0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, -0.47308734787878004, 0.0, 0.62583573544917614], [0.28209479177387814, 0.0, 0.0, 0.48860251190291992, 0.0, 0.0, -0.31539156525252005, 0.0, 0.54627421529603959, 0.0, 0.0, 0.0, 0.0, -0.45704579946446577, 0.0, 0.59004358992664352, 0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, -0.47308734787878004, 0.0, 0.62583573544917614], [0.28209479177387814, -0.48860251190291992, 0.0, 0.0, 0.0, 0.0, -0.31539156525252005, 0.0, -0.54627421529603959, 0.59004358992664352, 0.0, 0.45704579946446577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, 0.47308734787878004, 0.0, 0.62583573544917614], [0.28209479177387814, 0.48860251190291992, 0.0, 0.0, 0.0, 0.0, -0.31539156525252005, 0.0, -0.54627421529603959, -0.59004358992664352, 0.0, -0.45704579946446577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, 0.47308734787878004, 0.0, 0.62583573544917614], [0.28209479177387814, 0.0, -0.48860251190291992, 0.0, 0.0, 0.0, 0.63078313050504009, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7463526651802308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84628437532163447, 0.0, 0.0, 0.0, 0.0], [0.28209479177387814, 0.0, 0.48860251190291992, 0.0, 0.0, 0.0, 0.63078313050504009, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7463526651802308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84628437532163447, 0.0, 0.0, 0.0, 0.0], [0.28209479177387814, -0.3454941494713355, -0.3454941494713355, 0.0, 0.0, 0.54627421529603959, 0.15769578262626005, 0.0, -0.27313710764801985, 0.2086119118163921, 0.0, -0.4847702761712262, 0.13193775767639854, 0.0, 0.51099273821664126, 0.0, 0.0, -0.44253269244498283, 0.0, 0.16726163588932241, -0.34380302747441421, 0.0, -0.59135918484847527, 0.0, 0.15645893386229406], [0.28209479177387814, -0.3454941494713355, 0.3454941494713355, 0.0, 0.0, -0.54627421529603959, 0.15769578262626005, 0.0, -0.27313710764801985, 0.2086119118163921, 0.0, -0.4847702761712262, -0.13193775767639854, 0.0, -0.51099273821664126, 0.0, 0.0, 0.44253269244498283, 0.0, -0.16726163588932241, -0.34380302747441421, 0.0, -0.59135918484847527, 0.0, 0.15645893386229406], [0.28209479177387814, 0.3454941494713355, -0.3454941494713355, 0.0, 0.0, -0.54627421529603959, 0.15769578262626005, 0.0, -0.27313710764801985, -0.2086119118163921, 0.0, 0.4847702761712262, 0.13193775767639854, 0.0, 0.51099273821664126, 0.0, 0.0, 0.44253269244498283, 0.0, -0.16726163588932241, -0.34380302747441421, 0.0, -0.59135918484847527, 0.0, 0.15645893386229406], [0.28209479177387814, 0.3454941494713355, 0.3454941494713355, 0.0, 0.0, 0.54627421529603959, 0.15769578262626005, 0.0, -0.27313710764801985, -0.2086119118163921, 0.0, 0.4847702761712262, -0.13193775767639854, 0.0, -0.51099273821664126, 0.0, 0.0, -0.44253269244498283, 0.0, 0.16726163588932241, -0.34380302747441421, 0.0, -0.59135918484847527, 0.0, 0.15645893386229406], [0.28209479177387814, 0.0, -0.3454941494713355, -0.3454941494713355, 0.0, 0.0, 0.15769578262626005, 0.54627421529603959, 0.27313710764801985, 0.0, 0.0, 0.0, 0.13193775767639854, -0.4847702761712262, -0.51099273821664126, -0.2086119118163921, 0.0, 0.0, 0.0, 0.0, -0.34380302747441421, 0.16726163588932241, 0.59135918484847527, 0.44253269244498283, 0.15645893386229406], [0.28209479177387814, 0.0, -0.3454941494713355, 0.3454941494713355, 0.0, 0.0, 0.15769578262626005, -0.54627421529603959, 0.27313710764801985, 0.0, 0.0, 0.0, 0.13193775767639854, 0.4847702761712262, -0.51099273821664126, 0.2086119118163921, 0.0, 0.0, 0.0, 0.0, -0.34380302747441421, -0.16726163588932241, 0.59135918484847527, -0.44253269244498283, 0.15645893386229406], [0.28209479177387814, 0.0, 0.3454941494713355, -0.3454941494713355, 0.0, 0.0, 0.15769578262626005, -0.54627421529603959, 0.27313710764801985, 0.0, 0.0, 0.0, -0.13193775767639854, -0.4847702761712262, 0.51099273821664126, -0.2086119118163921, 0.0, 0.0, 0.0, 0.0, -0.34380302747441421, -0.16726163588932241, 0.59135918484847527, -0.44253269244498283, 0.15645893386229406], [0.28209479177387814, 0.0, 0.3454941494713355, 0.3454941494713355, 0.0, 0.0, 0.15769578262626005, 0.54627421529603959, 0.27313710764801985, 0.0, 0.0, 0.0, -0.13193775767639854, 0.4847702761712262, 0.51099273821664126, 0.2086119118163921, 0.0, 0.0, 0.0, 0.0, -0.34380302747441421, 0.16726163588932241, 0.59135918484847527, 0.44253269244498283, 0.15645893386229406], [0.28209479177387814, -0.3454941494713355, 0.0, -0.3454941494713355, 0.54627421529603959, 0.0, -0.3153915652525201, 0.0, 0.0, -0.41722382363278415, 0.0, 0.32318018411415073, 0.0, 0.32318018411415073, 0.0, 0.41722382363278415, -1.1102230246251565e-16, 0.0, -0.47308734787878015, 0.0, 0.31735664074561298, 0.0, 0.0, 0.0, -0.62583573544917659], [0.28209479177387814, 0.3454941494713355, 0.0, -0.3454941494713355, -0.54627421529603959, 0.0, -0.3153915652525201, 0.0, 0.0, 0.41722382363278415, 0.0, -0.32318018411415073, 0.0, 0.32318018411415073, 0.0, 0.41722382363278415, 1.1102230246251565e-16, 0.0, 0.47308734787878015, 0.0, 0.31735664074561298, 0.0, 0.0, 0.0, -0.62583573544917659], [0.28209479177387814, -0.3454941494713355, 0.0, 0.3454941494713355, -0.54627421529603959, 0.0, -0.3153915652525201, 0.0, 0.0, -0.41722382363278415, 0.0, 0.32318018411415073, 0.0, -0.32318018411415073, 0.0, -0.41722382363278415, 1.1102230246251565e-16, 0.0, 0.47308734787878015, 0.0, 0.31735664074561298, 0.0, 0.0, 0.0, -0.62583573544917659], [0.28209479177387814, 0.3454941494713355, 0.0, 0.3454941494713355, 0.54627421529603959, 0.0, -0.3153915652525201, 0.0, 0.0, 0.41722382363278415, 0.0, -0.32318018411415073, 0.0, -0.32318018411415073, 0.0, -0.41722382363278415, -1.1102230246251565e-16, 0.0, -0.47308734787878015, 0.0, 0.31735664074561298, 0.0, 0.0, 0.0, -0.62583573544917659], [0.28209479177387814, -0.28209479177387814, -0.28209479177387814, -0.28209479177387814, 0.36418281019735971, 0.36418281019735971, 0.0, 0.36418281019735971, 0.0, -0.22710788365184043, -0.55629843151037861, -0.17591701023519801, 0.28727127476813386, -0.17591701023519801, 0.0, 0.22710788365184048, 0.0, 0.3933623932844289, 0.42052208700336002, -0.14867700967939762, -0.32911059040285784, -0.14867700967939762, 0.0, -0.3933623932844289, -0.27814921575518947], [0.28209479177387814, -0.28209479177387814, 0.28209479177387814, -0.28209479177387814, 0.36418281019735971, -0.36418281019735971, 0.0, -0.36418281019735971, 0.0, -0.22710788365184043, 0.55629843151037861, -0.17591701023519801, -0.28727127476813386, -0.17591701023519801, 0.0, 0.22710788365184048, 0.0, -0.3933623932844289, 0.42052208700336002, 0.14867700967939762, -0.32911059040285784, 0.14867700967939762, 0.0, 0.3933623932844289, -0.27814921575518947], [0.28209479177387814, 0.28209479177387814, -0.28209479177387814, -0.28209479177387814, -0.36418281019735971, -0.36418281019735971, 0.0, 0.36418281019735971, 0.0, 0.22710788365184043, 0.55629843151037861, 0.17591701023519801, 0.28727127476813386, -0.17591701023519801, 0.0, 0.22710788365184048, 0.0, -0.3933623932844289, -0.42052208700336002, 0.14867700967939762, -0.32911059040285784, -0.14867700967939762, 0.0, -0.3933623932844289, -0.27814921575518947], [0.28209479177387814, 0.28209479177387814, 0.28209479177387814, -0.28209479177387814, -0.36418281019735971, 0.36418281019735971, 0.0, -0.36418281019735971, 0.0, 0.22710788365184043, -0.55629843151037861, 0.17591701023519801, -0.28727127476813386, -0.17591701023519801, 0.0, 0.22710788365184048, 0.0, 0.3933623932844289, -0.42052208700336002, -0.14867700967939762, -0.32911059040285784, 0.14867700967939762, 0.0, 0.3933623932844289, -0.27814921575518947], [0.28209479177387814, -0.28209479177387814, -0.28209479177387814, 0.28209479177387814, -0.36418281019735971, 0.36418281019735971, 0.0, -0.36418281019735971, 0.0, -0.22710788365184043, 0.55629843151037861, -0.17591701023519801, 0.28727127476813386, 0.17591701023519801, 0.0, -0.22710788365184048, 0.0, 0.3933623932844289, -0.42052208700336002, -0.14867700967939762, -0.32911059040285784, 0.14867700967939762, 0.0, 0.3933623932844289, -0.27814921575518947], [0.28209479177387814, -0.28209479177387814, 0.28209479177387814, 0.28209479177387814, -0.36418281019735971, -0.36418281019735971, 0.0, 0.36418281019735971, 0.0, -0.22710788365184043, -0.55629843151037861, -0.17591701023519801, -0.28727127476813386, 0.17591701023519801, 0.0, -0.22710788365184048, 0.0, -0.3933623932844289, -0.42052208700336002, 0.14867700967939762, -0.32911059040285784, -0.14867700967939762, 0.0, -0.3933623932844289, -0.27814921575518947], [0.28209479177387814, 0.28209479177387814, -0.28209479177387814, 0.28209479177387814, 0.36418281019735971, -0.36418281019735971, 0.0, -0.36418281019735971, 0.0, 0.22710788365184043, -0.55629843151037861, 0.17591701023519801, 0.28727127476813386, 0.17591701023519801, 0.0, -0.22710788365184048, 0.0, -0.3933623932844289, 0.42052208700336002, 0.14867700967939762, -0.32911059040285784, 0.14867700967939762, 0.0, 0.3933623932844289, -0.27814921575518947], [0.28209479177387814, 0.28209479177387814, 0.28209479177387814, 0.28209479177387814, 0.36418281019735971, 0.36418281019735971, 0.0, 0.36418281019735971, 0.0, 0.22710788365184043, 0.55629843151037861, 0.17591701023519801, -0.28727127476813386, 0.17591701023519801, 0.0, -0.22710788365184048, 0.0, 0.3933623932844289, 0.42052208700336002, -0.14867700967939762, -0.32911059040285784, -0.14867700967939762, 0.0, -0.3933623932844289, -0.27814921575518947], [0.28209479177387814, -0.14731920032792212, -0.14731920032792212, -0.44195760098376641, 0.29796775379783974, 0.099322584599279895, -0.22937568382001461, 0.29796775379783974, 0.39729033839711975, -0.42050234001046305, -0.23769603892429697, 0.07516608738008182, 0.28640664895524026, 0.22549826214024549, -0.31692805189906265, -0.29111700462262841, 0.49653083143075138, 0.38035867780395194, -0.093835507017278719, -0.14376206721065715, 0.05944972884490831, -0.43128620163197151, -0.12511400935637168, 0.26332523847965916, 0.14482149250063592], [0.28209479177387814, -0.14731920032792212, 0.14731920032792212, -0.44195760098376641, 0.29796775379783974, -0.099322584599279895, -0.22937568382001461, -0.29796775379783974, 0.39729033839711975, -0.42050234001046305, 0.23769603892429697, 0.07516608738008182, -0.28640664895524026, 0.22549826214024549, 0.31692805189906265, -0.29111700462262841, 0.49653083143075138, -0.38035867780395194, -0.093835507017278719, 0.14376206721065715, 0.05944972884490831, 0.43128620163197151, -0.12511400935637168, -0.26332523847965916, 0.14482149250063592], [0.28209479177387814, 0.14731920032792212, -0.14731920032792212, -0.44195760098376641, -0.29796775379783974, -0.099322584599279895, -0.22937568382001461, 0.29796775379783974, 0.39729033839711975, 0.42050234001046305, 0.23769603892429697, -0.07516608738008182, 0.28640664895524026, 0.22549826214024549, -0.31692805189906265, -0.29111700462262841, -0.49653083143075138, -0.38035867780395194, 0.093835507017278719, 0.14376206721065715, 0.05944972884490831, -0.43128620163197151, -0.12511400935637168, 0.26332523847965916, 0.14482149250063592], [0.28209479177387814, 0.14731920032792212, 0.14731920032792212, -0.44195760098376641, -0.29796775379783974, 0.099322584599279895, -0.22937568382001461, -0.29796775379783974, 0.39729033839711975, 0.42050234001046305, -0.23769603892429697, -0.07516608738008182, -0.28640664895524026, 0.22549826214024549, 0.31692805189906265, -0.29111700462262841, -0.49653083143075138, 0.38035867780395194, 0.093835507017278719, -0.14376206721065715, 0.05944972884490831, 0.43128620163197151, -0.12511400935637168, -0.26332523847965916, 0.14482149250063592], [0.28209479177387814, -0.14731920032792212, -0.14731920032792212, 0.44195760098376641, -0.29796775379783974, 0.099322584599279895, -0.22937568382001461, -0.29796775379783974, 0.39729033839711975, -0.42050234001046305, 0.23769603892429697, 0.07516608738008182, 0.28640664895524026, -0.22549826214024549, -0.31692805189906265, 0.29111700462262841, -0.49653083143075138, 0.38035867780395194, 0.093835507017278719, -0.14376206721065715, 0.05944972884490831, 0.43128620163197151, -0.12511400935637168, -0.26332523847965916, 0.14482149250063592], [0.28209479177387814, -0.14731920032792212, 0.14731920032792212, 0.44195760098376641, -0.29796775379783974, -0.099322584599279895, -0.22937568382001461, 0.29796775379783974, 0.39729033839711975, -0.42050234001046305, -0.23769603892429697, 0.07516608738008182, -0.28640664895524026, -0.22549826214024549, 0.31692805189906265, 0.29111700462262841, -0.49653083143075138, -0.38035867780395194, 0.093835507017278719, 0.14376206721065715, 0.05944972884490831, -0.43128620163197151, -0.12511400935637168, 0.26332523847965916, 0.14482149250063592], [0.28209479177387814, 0.14731920032792212, -0.14731920032792212, 0.44195760098376641, 0.29796775379783974, -0.099322584599279895, -0.22937568382001461, -0.29796775379783974, 0.39729033839711975, 0.42050234001046305, -0.23769603892429697, -0.07516608738008182, 0.28640664895524026, -0.22549826214024549, -0.31692805189906265, 0.29111700462262841, 0.49653083143075138, -0.38035867780395194, -0.093835507017278719, 0.14376206721065715, 0.05944972884490831, 0.43128620163197151, -0.12511400935637168, -0.26332523847965916, 0.14482149250063592], [0.28209479177387814, 0.14731920032792212, 0.14731920032792212, 0.44195760098376641, 0.29796775379783974, 0.099322584599279895, -0.22937568382001461, 0.29796775379783974, 0.39729033839711975, 0.42050234001046305, 0.23769603892429697, -0.07516608738008182, -0.28640664895524026, -0.22549826214024549, 0.31692805189906265, 0.29111700462262841, 0.49653083143075138, 0.38035867780395194, -0.093835507017278719, -0.14376206721065715, 0.05944972884490831, -0.43128620163197151, -0.12511400935637168, 0.26332523847965916, 0.14482149250063592], [0.28209479177387814, -0.44195760098376641, -0.14731920032792212, -0.14731920032792212, 0.29796775379783974, 0.29796775379783974, -0.22937568382001461, 0.099322584599279895, -0.39729033839711975, 0.29111700462262841, -0.23769603892429697, 0.22549826214024549, 0.28640664895524026, 0.07516608738008182, 0.31692805189906265, 0.42050234001046305, -0.49653083143075133, -0.26332523847965916, -0.093835507017278746, -0.43128620163197157, 0.059449728844908303, -0.14376206721065715, 0.12511400935637171, -0.38035867780395199, 0.14482149250063592], [0.28209479177387814, -0.44195760098376641, -0.14731920032792212, 0.14731920032792212, -0.29796775379783974, 0.29796775379783974, -0.22937568382001461, -0.099322584599279895, -0.39729033839711975, 0.29111700462262841, 0.23769603892429697, 0.22549826214024549, 0.28640664895524026, -0.07516608738008182, 0.31692805189906265, -0.42050234001046305, 0.49653083143075133, -0.26332523847965916, 0.093835507017278746, -0.43128620163197157, 0.059449728844908303, 0.14376206721065715, 0.12511400935637171, 0.38035867780395199, 0.14482149250063592], [0.28209479177387814, -0.44195760098376641, 0.14731920032792212, -0.14731920032792212, 0.29796775379783974, -0.29796775379783974, -0.22937568382001461, -0.099322584599279895, -0.39729033839711975, 0.29111700462262841, 0.23769603892429697, 0.22549826214024549, -0.28640664895524026, 0.07516608738008182, -0.31692805189906265, 0.42050234001046305, -0.49653083143075133, 0.26332523847965916, -0.093835507017278746, 0.43128620163197157, 0.059449728844908303, 0.14376206721065715, 0.12511400935637171, 0.38035867780395199, 0.14482149250063592], [0.28209479177387814, -0.44195760098376641, 0.14731920032792212, 0.14731920032792212, -0.29796775379783974, -0.29796775379783974, -0.22937568382001461, 0.099322584599279895, -0.39729033839711975, 0.29111700462262841, -0.23769603892429697, 0.22549826214024549, -0.28640664895524026, -0.07516608738008182, -0.31692805189906265, -0.42050234001046305, 0.49653083143075133, 0.26332523847965916, 0.093835507017278746, 0.43128620163197157, 0.059449728844908303, -0.14376206721065715, 0.12511400935637171, -0.38035867780395199, 0.14482149250063592], [0.28209479177387814, 0.44195760098376641, -0.14731920032792212, -0.14731920032792212, -0.29796775379783974, -0.29796775379783974, -0.22937568382001461, 0.099322584599279895, -0.39729033839711975, -0.29111700462262841, 0.23769603892429697, -0.22549826214024549, 0.28640664895524026, 0.07516608738008182, 0.31692805189906265, 0.42050234001046305, 0.49653083143075133, 0.26332523847965916, 0.093835507017278746, 0.43128620163197157, 0.059449728844908303, -0.14376206721065715, 0.12511400935637171, -0.38035867780395199, 0.14482149250063592], [0.28209479177387814, 0.44195760098376641, -0.14731920032792212, 0.14731920032792212, 0.29796775379783974, -0.29796775379783974, -0.22937568382001461, -0.099322584599279895, -0.39729033839711975, -0.29111700462262841, -0.23769603892429697, -0.22549826214024549, 0.28640664895524026, -0.07516608738008182, 0.31692805189906265, -0.42050234001046305, -0.49653083143075133, 0.26332523847965916, -0.093835507017278746, 0.43128620163197157, 0.059449728844908303, 0.14376206721065715, 0.12511400935637171, 0.38035867780395199, 0.14482149250063592], [0.28209479177387814, 0.44195760098376641, 0.14731920032792212, -0.14731920032792212, -0.29796775379783974, 0.29796775379783974, -0.22937568382001461, -0.099322584599279895, -0.39729033839711975, -0.29111700462262841, -0.23769603892429697, -0.22549826214024549, -0.28640664895524026, 0.07516608738008182, -0.31692805189906265, 0.42050234001046305, 0.49653083143075133, -0.26332523847965916, 0.093835507017278746, -0.43128620163197157, 0.059449728844908303, 0.14376206721065715, 0.12511400935637171, 0.38035867780395199, 0.14482149250063592], [0.28209479177387814, 0.44195760098376641, 0.14731920032792212, 0.14731920032792212, 0.29796775379783974, 0.29796775379783974, -0.22937568382001461, 0.099322584599279895, -0.39729033839711975, -0.29111700462262841, 0.23769603892429697, -0.22549826214024549, -0.28640664895524026, -0.07516608738008182, -0.31692805189906265, -0.42050234001046305, -0.49653083143075133, -0.26332523847965916, -0.093835507017278746, -0.43128620163197157, 0.059449728844908303, -0.14376206721065715, 0.12511400935637171, -0.38035867780395199, 0.14482149250063592], [0.28209479177387814, -0.14731920032792212, -0.44195760098376641, -0.14731920032792212, 0.099322584599279895, 0.29796775379783974, 0.45875136764002922, 0.29796775379783974, 0.0, -0.032346333846958689, -0.23769603892429694, -0.4259411618204636, -0.36823712008530907, -0.4259411618204636, 0.0, 0.032346333846958689, 3.4694469519536142e-18, 0.087775079493219665, 0.40662053040820756, 0.49763792495996717, 0.19933144377410417, 0.49763792495996717, 0.0, -0.087775079493219665, -0.020688784642947957], [0.28209479177387814, 0.14731920032792212, -0.44195760098376641, -0.14731920032792212, -0.099322584599279895, -0.29796775379783974, 0.45875136764002922, 0.29796775379783974, 0.0, 0.032346333846958689, 0.23769603892429694, 0.4259411618204636, -0.36823712008530907, -0.4259411618204636, 0.0, 0.032346333846958689, -3.4694469519536142e-18, -0.087775079493219665, -0.40662053040820756, -0.49763792495996717, 0.19933144377410417, 0.49763792495996717, 0.0, -0.087775079493219665, -0.020688784642947957], [0.28209479177387814, -0.14731920032792212, -0.44195760098376641, 0.14731920032792212, -0.099322584599279895, 0.29796775379783974, 0.45875136764002922, -0.29796775379783974, 0.0, -0.032346333846958689, 0.23769603892429694, -0.4259411618204636, -0.36823712008530907, 0.4259411618204636, 0.0, -0.032346333846958689, -3.4694469519536142e-18, 0.087775079493219665, -0.40662053040820756, 0.49763792495996717, 0.19933144377410417, -0.49763792495996717, 0.0, 0.087775079493219665, -0.020688784642947957], [0.28209479177387814, 0.14731920032792212, -0.44195760098376641, 0.14731920032792212, 0.099322584599279895, -0.29796775379783974, 0.45875136764002922, -0.29796775379783974, 0.0, 0.032346333846958689, -0.23769603892429694, 0.4259411618204636, -0.36823712008530907, 0.4259411618204636, 0.0, -0.032346333846958689, 3.4694469519536142e-18, -0.087775079493219665, 0.40662053040820756, -0.49763792495996717, 0.19933144377410417, -0.49763792495996717, 0.0, 0.087775079493219665, -0.020688784642947957], [0.28209479177387814, -0.14731920032792212, 0.44195760098376641, -0.14731920032792212, 0.099322584599279895, -0.29796775379783974, 0.45875136764002922, -0.29796775379783974, 0.0, -0.032346333846958689, 0.23769603892429694, -0.4259411618204636, 0.36823712008530907, -0.4259411618204636, 0.0, 0.032346333846958689, 3.4694469519536142e-18, -0.087775079493219665, 0.40662053040820756, -0.49763792495996717, 0.19933144377410417, -0.49763792495996717, 0.0, 0.087775079493219665, -0.020688784642947957], [0.28209479177387814, 0.14731920032792212, 0.44195760098376641, -0.14731920032792212, -0.099322584599279895, 0.29796775379783974, 0.45875136764002922, -0.29796775379783974, 0.0, 0.032346333846958689, -0.23769603892429694, 0.4259411618204636, 0.36823712008530907, -0.4259411618204636, 0.0, 0.032346333846958689, -3.4694469519536142e-18, 0.087775079493219665, -0.40662053040820756, 0.49763792495996717, 0.19933144377410417, -0.49763792495996717, 0.0, 0.087775079493219665, -0.020688784642947957], [0.28209479177387814, -0.14731920032792212, 0.44195760098376641, 0.14731920032792212, -0.099322584599279895, -0.29796775379783974, 0.45875136764002922, 0.29796775379783974, 0.0, -0.032346333846958689, -0.23769603892429694, -0.4259411618204636, 0.36823712008530907, 0.4259411618204636, 0.0, -0.032346333846958689, -3.4694469519536142e-18, -0.087775079493219665, -0.40662053040820756, -0.49763792495996717, 0.19933144377410417, 0.49763792495996717, 0.0, -0.087775079493219665, -0.020688784642947957], [0.28209479177387814, 0.14731920032792212, 0.44195760098376641, 0.14731920032792212, 0.099322584599279895, 0.29796775379783974, 0.45875136764002922, 0.29796775379783974, 0.0, 0.032346333846958689, 0.23769603892429694, 0.4259411618204636, 0.36823712008530907, 0.4259411618204636, 0.0, -0.032346333846958689, 3.4694469519536142e-18, 0.087775079493219665, 0.40662053040820756, 0.49763792495996717, 0.19933144377410417, 0.49763792495996717, 0.0, -0.087775079493219665, -0.020688784642947957]])
R_nv = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, -0.70710678118654757, -0.70710678118654757], [0.0, -0.70710678118654757, 0.70710678118654757], [0.0, 0.70710678118654757, -0.70710678118654757], [0.0, 0.70710678118654757, 0.70710678118654757], [-0.70710678118654757, 0.0, -0.70710678118654757], [0.70710678118654757, 0.0, -0.70710678118654757], [-0.70710678118654757, 0.0, 0.70710678118654757], [0.70710678118654757, 0.0, 0.70710678118654757], [-0.70710678118654757, -0.70710678118654757, 0.0], [-0.70710678118654757, 0.70710678118654757, 0.0], [0.70710678118654757, -0.70710678118654757, 0.0], [0.70710678118654757, 0.70710678118654757, 0.0], [-0.57735026918962573, -0.57735026918962573, -0.57735026918962573], [-0.57735026918962573, -0.57735026918962573, 0.57735026918962573], [-0.57735026918962573, 0.57735026918962573, -0.57735026918962573], [-0.57735026918962573, 0.57735026918962573, 0.57735026918962573], [0.57735026918962573, -0.57735026918962573, -0.57735026918962573], [0.57735026918962573, -0.57735026918962573, 0.57735026918962573], [0.57735026918962573, 0.57735026918962573, -0.57735026918962573], [0.57735026918962573, 0.57735026918962573, 0.57735026918962573], [-0.90453403373329089, -0.30151134457776357, -0.30151134457776357], [-0.90453403373329089, -0.30151134457776357, 0.30151134457776357], [-0.90453403373329089, 0.30151134457776357, -0.30151134457776357], [-0.90453403373329089, 0.30151134457776357, 0.30151134457776357], [0.90453403373329089, -0.30151134457776357, -0.30151134457776357], [0.90453403373329089, -0.30151134457776357, 0.30151134457776357], [0.90453403373329089, 0.30151134457776357, -0.30151134457776357], [0.90453403373329089, 0.30151134457776357, 0.30151134457776357], [-0.30151134457776357, -0.90453403373329089, -0.30151134457776357], [0.30151134457776357, -0.90453403373329089, -0.30151134457776357], [-0.30151134457776357, -0.90453403373329089, 0.30151134457776357], [0.30151134457776357, -0.90453403373329089, 0.30151134457776357], [-0.30151134457776357, 0.90453403373329089, -0.30151134457776357], [0.30151134457776357, 0.90453403373329089, -0.30151134457776357], [-0.30151134457776357, 0.90453403373329089, 0.30151134457776357], [0.30151134457776357, 0.90453403373329089, 0.30151134457776357], [-0.30151134457776357, -0.30151134457776357, -0.90453403373329089], [-0.30151134457776357, 0.30151134457776357, -0.90453403373329089], [0.30151134457776357, -0.30151134457776357, -0.90453403373329089], [0.30151134457776357, 0.30151134457776357, -0.90453403373329089], [-0.30151134457776357, -0.30151134457776357, 0.90453403373329089], [-0.30151134457776357, 0.30151134457776357, 0.90453403373329089], [0.30151134457776357, -0.30151134457776357, 0.90453403373329089], [0.30151134457776357, 0.30151134457776357, 0.90453403373329089]])

