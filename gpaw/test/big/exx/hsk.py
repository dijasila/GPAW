"""Collection of bulk systems.

From this paper:

  Judith Harl, Laurids Schimka, and Georg Kresse

  Assessing the quality of the random phase approximation for lattice
  constants and atomization energies of solids

  PHYSICAL REVIEW B 81, 115126 (2010)
"""

from ase.lattice import bulk
from ase.tasks.bulk import BulkTask

from gpaw.xc.hybridk import HybridXC


class HarlSchimkaKresseBulkCollection:
    data = {
        'C': ['diamond', 3.569, 3.534, 3.540, 3.572, 3.578, 3.553,
              7.72, 9.01, 5.18, 7.00, 6.93, 7.55],
        'Si': ['diamond', 5.466, 5.404, 5.482, 5.432, 5.445, 5.421,
               4.55, 5.34, 2.82, 4.39, 4.33, 4.68],
        'Ge': ['diamond', 5.765, 5.627, 5.701, 5.661, 5.676, 5.644,
               3.71, 4.62, 1.95, 3.59, 3.53, 3.92],
        'SiC': ['zincblende', 4.378, 4.332, 4.351, 4.365, 4.374, 4.346,
                6.40, 7.45, 4.36, 6.04, 5.96, 6.48],
        'AlN': ['zincblende', 4.397, 4.344, 4.346, 4.394, 4.402, 4.368,
                5.72, 6.68, 3.65, 5.46, 5.39, 5.85],
        'AlP': ['zincblende', 5.501, 5.435, 5.513, 5.467, 5.479, 5.451,
                4.09, 4.87, 2.53, 4.07, 4.02, 4.32],
        'AlAs': ['zincblende', 5.727, 5.630, 5.698, 5.675, 5.690, 5.649,
                 3.69, 4.52, 2.15, 3.67, 3.61, 3.82],
        'GaN': ['zincblende', 4.551, 4.460, 4.485, 4.519, 4.528, 4.520,
                4.37, 5.46, 2.22, 4.23, 4.17, 4.55],
        'GaP': ['zincblende', 5.507, 5.396, 5.519, 5.442, 5.456, 5.439,
                3.48, 4.39, 1.83, 3.48, 3.46, 3.61],
        'GaAs': ['zincblende', 5.749, 5.611, 5.706, 5.661, 5.675, 5.640,
                 3.14, 4.09, 1.51, 3.14, 3.09, 3.34],
        'InP': ['zincblende', 5.955, 5.827, 5.940, 5.867, 5.882, 5.858,
                3.14, 4.15, 1.56, 3.12, 3.07, 3.47],
        'InAs': ['zincblende', 6.184, 6.029, 6.120, 6.070, 6.087, 6.047,
                 2.88, 3.80, 1.31, 2.85, 2.80, 3.08],
        'InSb': ['zincblende', 6.633, 6.452, 6.585, 6.494, 6.514, 6.468,
                 2.63, 3.50, 1.10, 2.59, 2.55, 2.81],
        'MgO': ['rocksalt', 4.259, 4.169, 4.173, 4.225, 4.233, 4.189,
                4.98, 5.88, 3.47, 4.91, 4.83, 5.20],
        'LiF': ['rocksalt', 4.069, 3.913, 3.991, 3.998, 4.010, 3.972,
                4.33, 4.94, 3.25, 4.20, 4.15, 4.46],
        'NaF': ['rocksalt', 4.707, 4.511, 4.614, 4.625, 4.635, 4.582,
                3.82, 4.38, 2.79, 3.77, 3.71, 3.97],
        'LiCl': ['rocksalt', 5.149, 4.967, 5.272, 5.074, 5.091, 5.070,
                 3.37, 3.83, 2.68, 3.36, 3.31, 3.59],
        'NaCl': ['rocksalt', 5.697, 5.469, 5.778, 5.588, 5.607, 5.569,
                 3.10, 3.50, 2.54, 3.15, 3.11, 3.34],
        'Na': ['bcc', 4.196, 4.056, 4.494, 4.182, 4.208, 4.214,
               1.08, 1.26, 0.23, 1.00, 0.98, 1.12],
        'Al': ['fcc', 4.035, 3.983, 4.104, 4.037, 4.052, 4.018,
               3.44, 4.04, 1.33, 3.22, 3.14, 3.43],
        'Cu': ['fcc', 3.634, 3.523, 3.968, 3.597, 3.606, 3.595,
               3.48, 4.55, 0.03, 3.36, 3.30, 3.52],
        'Rh': ['fcc', 3.824, 3.753, 3.748, 3.811, 3.819, 3.794,
               5.74, 7.67, -2.88, 5.05, 4.97, 5.78],
        'Pd': ['fcc', 3.935, 3.830, 4.003, 3.896, 3.905, 3.876,
               3.74, 5.08, -1.26, 3.41, 3.35, 3.94],
        'Ag': ['fcc', 4.146, 4.002, 4.507, 4.087, 4.098, 4.062,
               2.52, 3.64, 0.52, 2.64, 2.58, 2.98]}
    
    names = ['C', 'Si', 'Ge', 'SiC', 'AlN', 'AlP', 'AlAs', 'GaN', 'GaP',
             'GaAs', 'InP', 'InAs', 'InSb', 'MgO', 'LiF', 'NaF', 'LiCl',
             'NaCl', 'Na', 'Al', 'Cu', 'Rh', 'Pd', 'Ag']

    def __init__(self, xc='PBE'):
        self.xc = xc

    def __getitem__(self, name):
        d = self.data[name]
        if self.xc == 'PBE':
            a = d[1]
        else:
            a = d[3]

        return bulk(name, crystalstructure=d[0], a=a)
    
    def keys(self):
        return self.names


class HarlSchimkaKresseBulkTask(BulkTask):
    def __init__(self, **kwargs):
        BulkTask.__init__(self,
                          collection=HarlSchimkaKresseBulkCollection('PBE'),
                          **kwargs)


class HarlSchimkaKresseEXXBulkTask(BulkTask):
    def __init__(self, **kwargs):
        BulkTask.__init__(self,
                          collection=HarlSchimkaKresseBulkCollection('EXX'),
                          **kwargs)

    def calculate(self, name, atoms):
        data = BulkTask.calculate(self, name, atoms)
        dexx = atoms.calc.get_xc_difference(HybridXC('EXX', acdf=True))
        data['selfconsistent energy'] = data['energy']
        data['energy'] += dexx
        return data


if __name__ == '__main__':
    from ase.tasks.main import run
    run(calcname='gpaw', task=HarlSchimkaKresseEXXBulkTask())
