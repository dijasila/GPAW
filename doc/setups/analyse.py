import os
import sys

import numpy as np
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
from ase.utils.eos import EquationOfState

from gpaw.test.big.setups.structures import fcc, rocksalt
from gpaw.atom.generator2 import get_number_of_electrons


sc = (chemical_symbols[3:5] + chemical_symbols[11:13] +
      chemical_symbols[19:31] + chemical_symbols[37:49] +
      chemical_symbols[55:81])

fcc = fcc()
rocksalt = rocksalt()

dir = sys.argv[1]

def get_data(symbol, mode, type, setup, x):
    name = '%s/%s/vol-%s-%s-%s-%s.dat' % (dir, symbol, mode, type, setup, x)
    try:
        data = np.array([float(f) for f in open(name)])
    except IOError:
        data = np.zeros(7) + np.nan
    return data


def analyse(data):
    vol = np.linspace(0.80, 1.1, 7)**3
    i = data.argmin()
    i2 = min(7, i + 2)
    i1 = i2 - 4
    v0 = e0 = np.nan
    if i1 > 0:
        eos = EquationOfState(vol[i1:i2], data[i1:i2])
        try:
            v0, e0, B = eos.fit()
        except (ValueError, np.linalg.LinAlgError):
            print 'XXX', i
    else:
        print 'XXX:', i
    x0 = v0**(1 / 3.0)
    return data[4], e0, x0, data[2] - data[4]


def run(symbol, setup, x):
    D = []
    print symbol,setup,x,
    if x == 'fcc':
        data = np.array(fcc.data[symbol])
    else:
        data = np.array(rocksalt.data[symbol])
    data = data[:-8:-1] + data[1]
    print 'aims',
    D.append(analyse(data))
    data = get_data(symbol, 'pw', 'nr', setup, x)
    print 'nr',
    D.append(analyse(data))
    data = get_data(symbol, 'pw', 'r', setup, x)
    print 'r',
    D.append(analyse(data))

    data = get_data(symbol, 'lcao', 'r', setup, x)
    print 'r-lcao',
    D.append(analyse(data))
    print
    return np.array(D)


def go(symbol, setup):
    f = run(symbol, setup, 'fcc')
    r = run(symbol, setup, 'rocksalt')
    
    name = '%s/%s/conv-pw-%s.dat' % (dir, symbol, setup)
    conv = np.array([float(x) for x in open(name)])
    name = '%s/%s/conv-atom-pw-%s.dat' % (dir, symbol, setup)
    conva = np.array([float(x) for x in open(name)])
    conv = np.array([conv, conva])
    
    egg = []
    for h in [0.2, 0.18, 0.16]:
        name = '%s/%s/egg-%s-%.2f.dat' % (dir, symbol, setup, h)
        try:
            egg.append(np.array([float(x.split()[0]) for x in open(name)]).ptp()*1000)
        except ValueError:
            egg.append(np.nan)
            
    ES=[]
    for es in ['rmm-diis', 'rmm-diis4']:
        name = '%s/%s/es-%s-nr-%s.dat' % (dir, symbol, es, setup)
        ES.append(float(open(name).readline()))

    return {'fcc': f, 'rocksalt': r, 'conv': conv, 'egg': np.array(egg),
            'es': np.array(ES)}


symbols = chemical_symbols[1:87]

S = {} 
for symbol in symbols:
    d = go(symbol, 'std')
    d['e'] = get_number_of_electrons(symbol, 'default')
    S[(symbol, 'std')] = d
    
for symbol in sc:
    d = go(symbol, 'sc')
    d['e'] = get_number_of_electrons(symbol, 'semicore')
    S[(symbol, 'sc')] = d


for s, t in S:
    d = S[(s, t)]
    print '%2s_%d,' % (s, d['e']),
    fe = d['fcc'][:, 1] + S[('O', 'std')]['fcc'][:, 1] - d['rocksalt'][:, 1]
    fe0 = fe[0]
    dfenr = fe[1] - fe[0]
    dferL = fe[3] - fe[2]
    
    arF0 = fcc.data[s][0]
    anrF0 = arF0 * d['fcc'][0, 2]
    anrF = arF0 * d['fcc'][1, 2]
    arF = arF0 * d['fcc'][2, 2]
    arFL = arF0 * d['fcc'][3, 2]
    
    arR0 = rocksalt.data[s][0]
    anrR0 = arR0 * d['rocksalt'][0, 2]
    anrR = arR0 * d['rocksalt'][1, 2]
    arR = arR0 * d['rocksalt'][2, 2]
    arRL = arR0 * d['rocksalt'][3, 2]
    
    ceF = d['fcc'][:, 3]
    dcenrF = ceF[1] - ceF[0]
    dcerFL = ceF[3] - ceF[2]
    
    ceR = d['rocksalt'][:, 3]
    dcenrR = ceR[1] - ceR[0]
    dcerRL = ceR[3] - ceR[2]

    eF, eA = d['conv']
    de = eF - d['fcc'][2, 0]
    dde = eA - eF
    dde -= dde[-1]
    
    egg = d['egg']
    
    es = d['es'] - eF[4]
    
    for x in ('arF0 anrF0 anrF arF arFL arR0 anrR0 anrR arR arRL ' +
              'fe0 dfenr dferL ceF dcenrF dcerFL ceR dcenrR dcerFL ' +
              'de dde egg es').split():
        d[x] = locals()[x]

import pickle
pickle.dump(S, open('data.pckl', 'w'))
