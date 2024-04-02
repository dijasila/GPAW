import pickle

g0w0 = 'MoS2_g0w0_80_results_GW.pckl'
g0w0g = 'MoS2_g0w0g_40_results_GWG.pckl'

res_g0w0 = pickle.load(open(g0w0, 'rb'), encoding='bytes')
res_g0w0g = pickle.load(open(g0w0g, 'rb'), encoding='bytes')


assert abs(res_g0w0['qp'][0, 0, 3:7] -
           [0.765, 2.248, 5.944, 5.944]).max() < 0.01
assert abs(res_g0w0g['qp'][0, 0, 3:7] -
           [1.158, 2.634, 6.407, 6.407]).max() < 0.01
