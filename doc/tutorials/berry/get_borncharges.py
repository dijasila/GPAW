import json
import numpy as np

with open('borncharges-0.01.json') as fd:
    Z_avv = json.load(fd)['Z_avv']
for a, Z_vv in enumerate(Z_avv):
    print(a, np.round(Z_vv, 2))
