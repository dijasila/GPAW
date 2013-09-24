from __future__ import division
import os
import sys

import numpy as np
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
from ase.utils.eos import EquationOfState

#from gpaw.test.big.setups.structures import fcc, rocksalt
from gpaw.atom.generator2 import get_number_of_electrons


sc = (chemical_symbols[3:5] + chemical_symbols[11:13] +
      chemical_symbols[19:31] + chemical_symbols[37:49] +
      chemical_symbols[55:57] + chemical_symbols[72:81])

fcc = np.loadtxt('fcc/energies_aims.csv',
                 delimiter=',',
                 skiprows=1,
                 usecols=range(1,25))
rocksalt = np.loadtxt('rocksalt/energies_aims.csv',
                      delimiter=',',
                      skiprows=1,
                      usecols=range(1,17))
volaims = np.array([0.8,0.85,0.9,0.925,0.95,0.975,1.0,1.025,1.05,1.075,
                    1.1,1.125,1.15,1.175,1.2,1.225,1.25,1.275,
                    1.3,1.325,1.35,1.375,1.4])**3

dir = sys.argv[1]

def get_data(symbol, mode, type, setup, x):
    name = '%s/%s/vol-%s-%s-%s-%s.dat' % (dir, symbol, mode, type, setup, x)
    try:
        data = np.array([float(f) for f in open(name)])
    except IOError:
        data = np.zeros(7) + np.nan
    return data


def analyse(data, vol, x1=None):
    i0 = data.argmin()
    i2 = min(len(data), i0 + 2)
    i1 = i2 - 4
    v0 = e0 = np.nan
    if i1 > 0:
        eos = EquationOfState(vol[i1:i2], data[i1:i2])
        try:
            v0, e0, B = eos.fit()
        except (ValueError, np.linalg.LinAlgError):
            print 'XXX', i0
    else:
        print 'XXX:', i0
    x0 = v0**(1 / 3.0)
    if x1 is None:
        i = i0
        x1 = np.nan
        while i > 0:
            if data[i] - e0 > 1.0:
                f = np.poly1d(np.polyfit(vol[i:i + 3],
                                         data[i:i + 3] - e0 - 1.0, 2))
                f1 = np.polyder(f, 1)
                for v1 in np.roots(f):
                    if f1(v1) < 0:
                        x1 = v1**(1 / 3)
                        break
                break
            i -= 1
        print i,x1,
    else:
        for i, v in enumerate(vol):
            if v > x1**3 and i > 0:
                f = np.poly1d(np.polyfit(vol[i - 1:i + 2], data[i - 1:i + 2] - e0 - 1.0, 2))
                x1 = f(x1**3)
                print x1,
                break
        else:
            x1 = np.nan
                
    return e0, x0, x1, data[4]


def run(symbol, setup, x):
    D = []
    print symbol,setup,x,
    Z = atomic_numbers[symbol]
    if x == 'fcc':
        data = fcc[Z - 1, 1:]
    else:
        data = rocksalt[Z - 1, 1:]
    print 'aims',
    D.append(analyse(data, volaims))
    x1 = D[0][2]
    
    data = get_data(symbol, 'pw', 'nr', setup, x)
    print 'nr',
    vol = np.linspace(0.80, 1.1, 7)**3
    D.append(analyse(data, vol, x1))
    
    data = get_data(symbol, 'pw', 'r', setup, x)
    print 'r',
    D.append(analyse(data, vol))
    x1 = D[2][2]
    
    data = get_data(symbol, 'lcao', 'r', setup, x)
    print 'r-lcao',
    D.append(analyse(data, vol, x1))
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
    Z = atomic_numbers[symbol]
    d = S[(s, t)]
    print '%2s_%d,' % (s, d['e']),
    fe = d['fcc'][:, 0] + S[('O', 'std')]['fcc'][:, 0] - d['rocksalt'][:, 0]
    fe0 = fe[0]
    dfenr = fe[1] - fe[0]
    dferL = fe[3] - fe[2]
    
    arF0 = fcc[Z - 1][0]
    anrF0 = arF0 * d['fcc'][0, 1]
    anrF = arF0 * d['fcc'][1, 1]
    arF = arF0 * d['fcc'][2, 1]
    arFL = arF0 * d['fcc'][3, 1]
    
    arR0 = rocksalt[Z - 1][0]
    anrR0 = arR0 * d['rocksalt'][0, 1]
    anrR = arR0 * d['rocksalt'][1, 1]
    arR = arR0 * d['rocksalt'][2, 1]
    arRL = arR0 * d['rocksalt'][3, 1]
    
    dceR = d['rocksalt'][:, 2]
    dcenrR = dceR[1]
    dcerRL = dceR[3]

    eF, eA = d['conv']
    de = eF - d['fcc'][2, 3]
    dde = eA - eF
    dde -= dde[-1]
    
    egg = d['egg']
    
    es = d['es'] - eF[4]
    
    for x in ('arF0 anrF0 anrF arF arFL arR0 anrR0 anrR arR arRL ' +
              'fe0 dfenr dferL dcenrR dcerRL ' +
              'de dde egg es').split():
        d[x] = locals()[x]

import pickle
pickle.dump(S, open('data.pckl', 'w'))
