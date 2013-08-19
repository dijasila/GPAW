from numpy import nan
from ase.lattice import bulk
from ase.data import chemical_symbols


class fcc:
    def __getitem__(self, name):
        a = fcc_data[name][0]
        return bulk(name, 'fcc', a=a)

    def keys(self):
        return chemical_symbols[1:103]


class rocksalt:
    def __getitem__(self, name):
        a = rocksalt_data[name][0]
        return bulk(name + 'O', 'rocksalt', a=a)

    def keys(self):
        return chemical_symbols[1:103]


fcc_data = {
    'H':  (2.28, -14.6321,
           0.0736, 0.0170, 0.0000, 0.0263, 0.1112, 0.2963, 0.5927),
    'He': (4.08, -78.7314,
           0.0016, 0.0002, 0.0000, 0.0025, 0.0105, 0.0288, 0.0651),
    'Li': (4.33, -204.6588,
           0.0653, 0.0176, 0.0000, 0.0250, 0.1031, 0.2516, 0.5259),
    'Be': (3.16, -401.7460,
           0.2111, 0.0587, 0.0000, 0.0675, 0.3334, 0.8715, 1.7946),
    'B':  (2.87, -674.6311,
           0.3576, 0.1050, 0.0000, 0.1223, 0.5876, 1.5282, 3.1691),
    'C':  (3.08, -1031.8329,
           0.2331, 0.0651, 0.0000, 0.0898, 0.4116, 1.0735, 2.2243),
    'N':  (3.12, -1484.6948,
           0.2746, 0.0769, 0.0000, 0.1004, 0.5319, 1.4115, 2.9923),
    'O':  (3.18, -2041.6291,
           0.2252, 0.0688, 0.0000, 0.0919, 0.4453, 1.2414, 2.7536),
    'F':  (3.43, -2712.3849,
           0.0974, 0.0303, 0.0000, 0.0461, 0.2383, 0.6910, 1.5892),
    'Ne': (4.58, -3506.6526,
           0.0042, 0.0012, 0.0000, 0.0037, 0.0185, 0.0566, 0.1421),
    'Na': (5.31, -4414.0216,
           0.0513, 0.0193, 0.0000, 0.0244, 0.1063, 0.2746, 0.5814),
    'Mg': (4.53, -5442.5517,
           0.1808, 0.0581, 0.0000, 0.0639, 0.3099, 0.8174, 1.7330),
    'Al': (4.04, -6595.0427,
           0.2656, 0.0973, 0.0000, 0.1112, 0.5292, 1.4284, 3.0616),
    'Si': (3.87, -7874.4993,
           0.2688, 0.0738, 0.0000, 0.1017, 0.4730, 1.3005, 2.9060),
    'P':  (3.88, -9284.8472,
           0.2874, 0.0817, 0.0000, 0.0974, 0.4914, 1.3413, 2.8181),
    'S':  (3.99, -10830.6086,
           0.2488, 0.0706, 0.0000, 0.1101, 0.5056, 1.2975, 2.8025),
    'Cl': (4.40, -12517.1110,
           0.1260, 0.0422, 0.0000, 0.0550, 0.2837, 0.8156, 1.8181),
    'Ar': (5.95, -14349.8398,
           0.0058, 0.0020, 0.0000, 0.0036, 0.0207, 0.0667, 0.1729),
    'K':  (6.65, -16319.8327,
           0.0709, 0.0322, 0.0000, 0.0244, 0.1062, 0.2924, 0.5927),
    'Ca': (5.53, -18433.5503,
           0.1553, 0.0370, 0.0000, 0.0542, 0.2503, 0.6447, 1.2979),
    'Sc': (4.62, -20695.6592,
           0.2878, 0.0803, 0.0000, 0.1081, 0.4438, 1.1173, 2.2134),
    'Ti': (4.11, -23111.3417,
           0.3993, 0.1155, 0.0000, 0.1491, 0.6724, 1.7014, 3.4002),
    'V':  (3.82, -25684.3238,
           0.4953, 0.1371, 0.0000, 0.1988, 0.9148, 2.3440, 4.7496),
    'Cr': (3.62, -28418.4359,
           0.5377, 0.1469, 0.0000, 0.2570, 1.1289, 2.9310, 6.0810),
    'Mn': (3.50, -31317.4164,
           0.5404, 0.1450, 0.0000, 0.3033, 1.2837, 3.3622, 7.1169),
    'Fe': (3.45, -34384.9782,
           0.5169, 0.1275, 0.0000, 0.2819, 1.2733, 3.3765, 7.2872),
    'Co': (3.46, -37625.0416,
           0.4663, 0.1204, 0.0000, 0.2525, 1.1497, 3.1114, 6.8072),
    'Ni': (3.51, -41041.5627,
           0.3623, 0.0854, 0.0000, 0.2435, 1.0440, 2.7725, 6.0897),
    'Cu': (3.63, -44638.1238,
           0.2399, 0.0479, 0.0000, 0.1918, 0.8292, 2.2403, 4.9274),
    'Zn': (3.94, -48415.1934,
           0.1715, 0.0430, 0.0000, 0.1091, 0.5070, 1.4055, 3.1477),
    'Ga': (4.23, -52373.0133,
           0.2214, 0.0553, 0.0000, 0.0906, 0.4085, 1.1288, 2.5802),
    'Ge': (4.28, -56512.2428,
           0.2404, 0.0731, 0.0000, 0.0885, 0.4687, 1.3615, 3.0379),
    'As': (4.26, -60834.3147,
           0.3211, 0.0894, 0.0000, 0.1323, 0.6272, 1.6811, 3.6754),
    'Se': (4.33, -65341.5130,
           0.2519, 0.0724, 0.0000, 0.1461, 0.6402, 1.6913, 3.6076),
    'Br': (4.73, -70036.9533,
           0.1225, 0.0354, 0.0000, 0.0661, 0.3384, 0.9661, 2.1901),
    'Kr': (6.31, -74924.2872,
           0.0037, 0.0001, 0.0000, 0.0085, 0.0363, 0.1047, 0.2566),
    'Rb': (7.19, -79994.7686,
           0.0489, 0.0124, 0.0000, 0.0251, 0.0788, 0.2333, 0.5380),
    'Sr': (6.04, -85253.6230,
           0.1433, 0.0421, 0.0000, 0.0425, 0.2240, 0.5763, 1.1235),
    'Y':  (5.06, -90703.7120,
           0.2938, 0.0792, 0.0000, 0.0998, 0.4232, 1.0884, 2.1513),
    'Zr': (4.53, -96348.1270,
           0.4695, 0.1425, 0.0000, 0.1511, 0.7068, 1.7900, 3.6718),
    'Nb': (4.21, -102188.9833,
           0.5852, 0.1532, 0.0000, 0.2624, 1.1388, 2.8633, 5.7619),
    'Mo': (4.00, -108228.6392,
           0.6767, 0.1691, 0.0000, 0.3318, 1.4846, 3.8471, 8.0495),
    'Tc': (3.87, -114469.2968,
           0.7072, 0.1698, 0.0000, 0.4424, 1.8754, 4.8154, 10.1020),
    'Ru': (3.81, -120913.2410,
           0.6325, 0.1387, 0.0000, 0.4627, 2.0001, 5.2035, 11.0653),
    'Rh': (3.83, -127562.9965,
           0.4415, 0.0819, 0.0000, 0.4482, 1.8415, 4.8459, 10.4745),
    'Pd': (3.94, -134421.3753,
           0.2259, 0.0178, 0.0000, 0.3758, 1.4724, 3.8336, 8.4586),
    'Ag': (4.14, -141490.0527,
           0.1106, -0.0037, 0.0000, 0.2773, 1.0763, 2.7872, 6.1386),
    'Cd': (4.50, -148767.0412,
           0.1362, 0.0291, 0.0000, 0.1453, 0.6176, 1.6535, 3.7345),
    'In': (4.80, -156252.8867,
           0.1957, 0.0503, 0.0000, 0.0546, 0.4391, 1.2738, 2.9924),
    'Sn': (4.82, -163948.4097,
           0.2763, 0.0772, 0.0000, 0.1260, 0.5959, 1.6293, 3.6919),
    'Sb': (4.79, -171854.4070,
           0.3068, 0.0858, 0.0000, 0.1178, 0.6626, 1.8997, 4.2431),
    'Te': (4.83, -179972.4842,
           0.2798, 0.0772, 0.0000, 0.1384, 0.6862, 1.9115, 4.1607),
    'I':  (5.20, -188304.5705,
           0.1270, 0.0338, 0.0000, 0.0871, 0.4048, 1.1161, 2.5094),
    'Xe': (6.69, -196853.5208,
           -0.0049, -0.0062, 0.0000, 0.0225, 0.0791, 0.2047, 0.4668),
    'Cs': (7.79, -205611.6193,
           0.0005, -0.0300, 0.0000, 0.0348, 0.1174, 0.2287, 0.4213),
    'Ba': (6.35, -214583.9127,
           0.1508, 0.0383, 0.0000, 0.0218, 0.1040, 0.2498, 0.4954),
    'La': (5.25, -223773.6480,
           0.2784, 0.1164, 0.0000, -0.0762, -0.0915, 0.0623, 0.7079),
    'Ce': (4.71, -233187.2627,
           0.3433, 0.1630, 0.0000, -0.0986, 0.0599, 0.7613, 2.4307),
    'Pr': (4.56, -242828.2889,
           0.3334, 0.1724, 0.0000, -0.0924, 0.1163, 0.9280, 2.7628),
    'Nd': (4.47, -252699.2744,
           0.3296, 0.1750, 0.0000, -0.0885, 0.1241, 0.9447, 2.8224),
    'Pm': (4.42, -262802.2912,
           0.2960, 0.1588, 0.0000, -0.0933, 0.1044, 0.8999, 2.7589),
    'Sm': (4.43, -273139.3306,
           0.2112, 0.1166, 0.0000, -0.0954, 0.0416, 0.7121, 2.3722),
    'Eu': (4.56, -283712.4427,
           0.0765, 0.0494, 0.0000, -0.0663, -0.0624, 0.2876, 1.3735),
    'Gd': (4.65, -294523.7409,
           -0.0515, -0.0229, 0.0000, 0.0009, 0.0168, 0.2157, 0.9975),
    'Tb': (4.86, -305575.3657,
           -0.1837, -0.0991, 0.0000, 0.1059, 0.1942, 0.3112, 0.6629),
    'Dy': (4.98, -316869.1953,
           -0.2253, -0.1449, 0.0000, 0.1690, 0.3573, 0.5515, 0.8386),
    'Ho': (5.13, -328407.2997,
           -0.2254, -0.1513, 0.0000, 0.2011, 0.4564, 0.7553, 1.0882),
    'Er': (5.18, -340191.5430,
           -0.2383, -0.1556, 0.0000, 0.2292, 0.5408, 0.9230, 1.3675),
    'Tm': (5.22, -352223.9747,
           -0.2415, -0.1633, 0.0000, 0.2507, 0.6127, 1.0884, 1.6964),
    'Yb': (5.24, -364505.9700,
           -0.1823, -0.1445, 0.0000, 0.2867, 0.7147, 1.3266, 2.1875),
    'Lu': (4.86, -377034.5126,
           0.1386, -0.0091, 0.0000, 0.2148, 0.6847, 1.5372, 2.9031),
    'Hf': (4.48, -389809.8009,
           0.3502, 0.0689, 0.0000, 0.2662, 0.9903, 2.3323, 4.6401),
    'Ta': (4.21, -402832.8243,
           0.4280, 0.0470, 0.0000, 0.4454, 1.6063, 3.7645, 7.3705),
    'W':  (4.03, -416104.8297,
           0.4864, 0.0321, 0.0000, 0.5840, 2.1661, 5.2155, 10.5427),
    'Re': (3.91, -429626.9743,
           0.4480, -0.0234, 0.0000, 0.8258, 2.9196, 6.9115, 13.9019),
    'Os': (3.85, -443400.5241,
           0.3094, -0.1039, 0.0000, 0.9551, 3.3739, 8.0261, 16.2023),
    'Ir': (3.87, -457427.0892,
           0.0912, -0.1705, 0.0000, 0.9548, 3.2939, 7.9219, 16.2156),
    'Pt': (3.97, -471708.6220,
           -0.1344, -0.2525, 0.0000, 0.8484, 2.8497, 6.8157, 14.1286),
    'Au': (4.16, -486246.1218,
           -0.1872, -0.2141, 0.0000, 0.6400, 2.1715, 5.1936, 10.8173),
    'Hg': (4.77, -501037.1317,
           0.1816, 0.0556, 0.0000, 0.0864, 0.4841, 1.4533, 3.5023),
    'Tl': (4.97, -516081.4987,
           0.1289, 0.0086, 0.0000, 0.1765, 0.6440, 1.7388, 3.8707),
    'Pb': (5.02, -531379.7968,
           0.1945, 0.0325, 0.0000, 0.1931, 0.8224, 2.1437, 4.6255),
    'Bi': (5.01, -546932.3838,
           0.2300, 0.0376, 0.0000, 0.2150, 0.8929, 2.3827, 5.1848),
    'Po': (5.06, -562740.3651,
           0.2331, 0.0546, 0.0000, 0.2016, 0.8704, 2.2806, 4.9617),
    'At': (5.36, -578805.0358,
           0.0806, -0.0030, 0.0000, 0.1583, 0.6094, 1.5514, 3.3525),
    'Rn': (6.75, -595128.6847,
           -0.0206, -0.0168, 0.0000, 0.0435, 0.1409, 0.3443, 0.7507),
    'Fr': (7.68, -611703.9670,
           -0.0667, -0.0673, 0.0000, 0.0563, 0.1315, 0.2753, 0.4208),
    'Ra': (6.52, -628535.6579,
           0.1319, 0.0405, 0.0000, -0.0111, 0.0192, 0.0865, 0.2019),
    'Ac': (5.64, -645626.0083,
           0.4893, 0.2164, 0.0000, -0.2038, -0.4149, -0.4310, -0.0962),
    'Th': (5.04, -662979.0156,
           0.8718, 0.4418, 0.0000, -0.3977, -0.5552, -0.2450, 0.9007),
    'Pa': (4.65, -680597.8097,
           0.9665, 0.4546, 0.0000, -0.2562, -0.1130, 0.7264, 2.7561),
    'U':  (4.41, -698484.1687,
           0.8914, 0.3879, 0.0000, -0.1112, 0.2989, 1.6657, 4.6282),
    'Np': (4.24, -716639.6871,
           0.7267, 0.2753, 0.0000, 0.0917, 0.8818, 2.8991, 6.9445),
    'Pu': (4.14, -735065.6866,
           0.4573, 0.1196, 0.0000, 0.3287, 1.4616, 4.0010, 8.8614),
    'Am': (4.10, -753763.7993,
           0.2041, -0.0210, 0.0000, 0.4829, 1.8184, 4.6166, 9.8188),
    'Cm': (4.09, -772735.6560,
           0.0101, -0.1185, 0.0000, 0.5966, 2.0509, 4.9509, 10.2339),
    'Bk': (4.15, -791982.8229,
           -0.0552, -0.1314, 0.0000, 0.5349, 1.8086, 4.3662, 9.0755),
    'Cf': (4.25, -811506.6988,
           -0.0806, -0.1168, 0.0000, 0.4277, 1.4462, 3.5130, 7.3963),
    'Es': (4.40, -831308.7076,
           -0.0858, -0.0943, 0.0000, 0.3051, 1.0312, 2.5446, 5.4913),
    'Fm': (4.62, -851390.2087,
           -0.1173, -0.0812, 0.0000, 0.2126, 0.6916, 1.6761, 3.6334),
    'Md': (4.89, -871750.1905,
           -0.6607, -0.3498, 0.0000, 0.3805, 0.7822, 1.4490, 2.7495),
    'No': (5.16, -892387.5835,
           -0.3969, -0.2508, 0.0000, 0.3580, 0.8653, 1.5453, 2.4901)}

rocksalt_data = {
    'H':  (3.42, -2058.0486,
           0.2431, 0.0739, 0.0000, 0.1065, 0.5207, 1.4563, 3.3311),
    'He': (4.85, -2118.3532,
           0.0472, 0.0153, 0.0000, 0.0234, 0.1130, 0.2890, 0.5694),
    'Li': (4.06, -2250.8730,
           0.2741, 0.0837, 0.0000, 0.1334, 0.6950, 2.1252, 5.4488),
    'Be': (3.65, -2450.7099,
           0.5834, 0.1750, 0.0000, 0.2418, 1.2597, 3.8001, 9.5556),
    'B':  (3.89, -2718.7593,
           0.4071, 0.1122, 0.0000, 0.1402, 0.6406, 1.9555, 5.4011),
    'C':  (3.97, -3075.9676,
           0.3105, 0.0795, 0.0000, 0.1303, 0.6329, 1.8908, 4.7957),
    'N':  (3.94, -3528.2353,
           0.4106, 0.1301, 0.0000, 0.2412, 1.1120, 2.8174, 5.4905),
    'O':  (3.99, -4084.9335,
           0.5080, 0.1653, 0.0000, 0.3012, 1.4834, 3.4570, 6.3440),
    'F':  (4.22, -4754.5968,
           0.3577, 0.1180, 0.0000, 0.2306, 1.1449, 2.9499, 5.5540),
    'Ne': (5.53, -5546.1716,
           0.0364, 0.0122, 0.0000, 0.0263, 0.1562, 0.5207, 1.2595),
    'Na': (4.80, -6459.3998,
           0.2564, 0.0797, 0.0000, 0.1270, 0.7183, 2.3426, 6.2241),
    'Mg': (4.26, -7492.3029,
           0.6260, 0.1927, 0.0000, 0.2786, 1.5109, 4.7735, 12.6731),
    'Al': (4.48, -8641.7538,
           0.5608, 0.1803, 0.0000, 0.2268, 1.2011, 3.5786, 9.2559),
    'Si': (4.61, -9919.9807,
           0.4953, 0.1499, 0.0000, 0.2414, 1.1169, 3.1669, 7.8148),
    'P':  (4.61, -11329.2669,
           0.6038, 0.1935, 0.0000, 0.2487, 1.2374, 3.3555, 7.6965),
    'S':  (4.62, -12874.1946,
           0.5582, 0.1780, 0.0000, 0.3136, 1.5078, 3.6935, 7.8564),
    'Cl': (4.76, -14559.7527,
           0.4863, 0.1591, 0.0000, 0.2797, 1.3855, 3.4075, 6.8490),
    'Ar': (5.37, -16389.7231,
           0.1212, 0.0448, 0.0000, 0.0925, 0.5071, 1.4601, 3.3949),
    'K':  (5.53, -18365.0658,
           0.2273, 0.0708, 0.0000, 0.1171, 0.6214, 1.7790, 4.3775),
    'Ca': (4.83, -20483.8405,
           0.6024, 0.1813, 0.0000, 0.2946, 1.5419, 4.7283, 11.7134),
    'Sc': (4.47, -22745.8729,
           0.8166, 0.2501, 0.0000, 0.4299, 2.2478, 6.7310, 16.1234),
    'Ti': (4.28, -25160.3821,
           0.8790, 0.2623, 0.0000, 0.4638, 2.4175, 7.3695, 18.2262),
    'V':  (4.18, -27731.8578,
           0.8167, 0.2343, 0.0000, 0.4431, 2.3640, 7.3525, 18.6324),
    'Cr': (4.13, -30464.6790,
           0.7671, 0.2261, 0.0000, 0.4219, 2.2428, 7.1135, 18.4056),
    'Mn': (4.10, -33362.8682,
           0.7137, 0.2028, 0.0000, 0.4069, 2.1557, 6.9165, 18.2496),
    'Fe': (4.09, -36430.2043,
           0.6735, 0.1931, 0.0000, 0.3746, 2.0397, 6.6940, 18.1184),
    'Co': (4.10, -39670.4123,
           0.6537, 0.1856, 0.0000, 0.4050, 2.1457, 6.8666, 18.2597),
    'Ni': (4.17, -43087.0107,
           0.7323, 0.2279, 0.0000, 0.3842, 2.0330, 6.4120, 16.8641),
    'Cu': (4.24, -46683.4078,
           0.6268, 0.1792, 0.0000, 0.4059, 2.0126, 6.2211, 16.1968),
    'Zn': (4.33, -50462.3439,
           0.6274, 0.1872, 0.0000, 0.3397, 1.7735, 5.5976, 15.0824),
    'Ga': (4.63, -54419.0933,
           0.5963, 0.1975, 0.0000, 0.2134, 1.1920, 3.8435, 10.6075),
    'Ge': (4.78, -58558.0961,
           0.5217, 0.1547, 0.0000, 0.2179, 1.1972, 3.6469, 9.5608),
    'As': (4.77, -62879.1790,
           0.5580, 0.1627, 0.0000, 0.3390, 1.5746, 4.4417, 10.5922),
    'Se': (4.84, -67385.3273,
           0.5649, 0.1710, 0.0000, 0.3453, 1.6523, 4.2781, 9.8491),
    'Br': (4.99, -72079.7951,
           0.4983, 0.1680, 0.0000, 0.3228, 1.5766, 3.9462, 8.5283),
    'Kr': (5.42, -76964.6261,
           0.1891, 0.0626, 0.0000, 0.1583, 0.8099, 2.2593, 5.4180),
    'Rb': (5.78, -82039.9498,
           0.1968, 0.0561, 0.0000, 0.1023, 0.5381, 1.7024, 4.5176),
    'Sr': (5.20, -87303.4991,
           0.5869, 0.1772, 0.0000, 0.2890, 1.5096, 4.6172, 11.2819),
    'Y':  (4.83, -92753.7832,
           0.8616, 0.2743, 0.0000, 0.4282, 2.3169, 6.9392, 16.5457),
    'Zr': (4.60, -98397.1865,
           0.9785, 0.2998, 0.0000, 0.5061, 2.7533, 8.3887, 20.4979),
    'Nb': (4.47, -104236.3355,
           0.9398, 0.2785, 0.0000, 0.5467, 2.8773, 8.9823, 22.4638),
    'Mo': (4.41, -110274.2271,
           0.8499, 0.2438, 0.0000, 0.5578, 2.8330, 8.9279, 22.8560),
    'Tc': (4.39, -116513.7364,
           0.7472, 0.1999, 0.0000, 0.5254, 2.7135, 8.6324, 22.5713),
    'Ru': (4.40, -122957.3301,
           0.6414, 0.1639, 0.0000, 0.4898, 2.5406, 8.1920, 21.9123),
    'Rh': (4.45, -129607.4081,
           0.6338, 0.1753, 0.0000, 0.4722, 2.4564, 7.8741, 20.8961),
    'Pd': (4.53, -136465.9337,
           0.6342, 0.1827, 0.0000, 0.5199, 2.5646, 7.8168, 19.9561),
    'Ag': (4.67, -143534.5420,
           0.5106, 0.1282, 0.0000, 0.4732, 2.2600, 6.9531, 17.9321),
    'Cd': (4.77, -150813.9190,
           0.6158, 0.1769, 0.0000, 0.3936, 2.0123, 6.3680, 17.2253),
    'In': (4.96, -158299.1950,
           0.5650, 0.1547, 0.0000, 0.2741, 1.5062, 4.9800, 13.9290),
    'Sn': (5.13, -165994.3672,
           0.5611, 0.1785, 0.0000, 0.2487, 1.2833, 4.0891, 11.3525),
    'Sb': (5.15, -173899.6548,
           0.6311, 0.1849, 0.0000, 0.3077, 1.5256, 4.5301, 11.6126),
    'Te': (5.20, -182016.6219,
           0.5830, 0.1719, 0.0000, 0.3621, 1.6135, 4.2877, 10.6561),
    'I':  (5.35, -190347.6505,
           0.5130, 0.1494, 0.0000, 0.3359, 1.5744, 3.8332, 8.9053),
    'Xe': (5.66, -198894.4554,
           0.2568, 0.0743, 0.0000, 0.2181, 1.0380, 2.7730, 6.3836),
    'Cs': (5.96, -207657.0159,
           0.2155, 0.0679, 0.0000, 0.1282, 0.6789, 2.0629, 5.2515),
    'Ba': (5.59, -216633.4387,
           0.5709, 0.1755, 0.0000, 0.2548, 1.3275, 3.8826, 9.0137),
    'La': (5.15, -225823.6974,
           0.6684, 0.1652, 0.0000, 0.5300, 2.3973, 6.4291, 13.9275),
    'Ce': (4.99, -235236.9754,
           0.4500, 0.0403, 0.0000, 0.7210, 2.8369, 7.1922, 15.2990),
    'Pr': (4.94, -244877.8809,
           0.3168, -0.0221, 0.0000, 0.7660, 2.8986, 7.2109, 15.1028),
    'Nd': (4.90, -254748.8280,
           0.2433, -0.0593, 0.0000, 0.7981, 2.9528, 7.2874, 15.1235),
    'Pm': (4.87, -264851.8568,
           0.1436, -0.1077, 0.0000, 0.8335, 3.0101, 7.3507, 15.2259),
    'Sm': (4.86, -275189.0111,
           nan, -0.1251, 0.0000, 0.8275, 2.9632, 7.2338, 15.0575),
    'Eu': (4.84, -285762.2937,
           -0.0202, -0.1577, 0.0000, 0.8542, 3.0214, 7.3623, 15.2954),
    'Gd': (4.82, -296573.7002,
           nan, -0.1843, 0.0000, 0.8783, 3.0797, 7.5147, 15.6221),
    'Tb': (4.83, -307625.3602,
           0.0606, -0.1856, 0.0000, 0.8385, 2.9449, 7.3126, 15.3526),
    'Dy': (4.83, -318919.1590,
           0.0456, -0.1778, 0.0000, 0.8353, 2.9284, 7.2737, 15.3656),
    'Ho': (4.82, -330457.1053,
           -0.0037, -0.1867, 0.0000, 0.8529, 2.9679, 7.3875, 15.6631),
    'Er': (4.81, -342241.2359,
           -0.0142, -0.1941, 0.0000, 0.8699, 3.0179, 7.5347, 16.0627),
    'Tm': (4.81, -354273.5595,
           -0.0068, -0.1895, 0.0000, 0.8709, 3.0281, 7.6225, 16.4065),
    'Yb': (4.83, -366556.0025,
           0.0227, -0.1736, 0.0000, 0.8480, 2.9814, nan, nan),
    'Lu': (4.74, -379084.5334,
           0.7011, 0.1724, 0.0000, 0.6208, 2.8279, 8.0913, 19.0624),
    'Hf': (4.59, -391858.7119,
           0.8599, 0.2254, 0.0000, 0.6390, 3.1480, 9.3513, 22.8034),
    'Ta': (4.51, -404879.9886,
           0.9104, 0.2490, 0.0000, 0.6127, 3.1264, 9.7091, 24.4307),
    'W':  (4.46, -418150.1359,
           0.7801, 0.1932, 0.0000, 0.6830, 3.2581, 10.0462, 25.7076),
    'Re': (4.45, -431671.0057,
           0.6499, 0.1274, 0.0000, 0.6899, 3.2571, 10.0351, 25.9527),
    'Os': (4.48, -445444.1489,
           0.5364, 0.0832, 0.0000, 0.6538, 3.0694, 9.5316, 25.0749),
    'Ir': (4.55, -459471.1125,
           0.5274, 0.0996, 0.0000, 0.6207, 2.9343, 9.0592, 23.6318),
    'Pt': (4.63, -473752.9823,
           0.4960, 0.0796, 0.0000, 0.7155, 3.1890, 9.3027, 23.0762),
    'Au': (4.75, -488290.4217,
           0.3057, -0.0133, 0.0000, 0.7726, 3.1436, 8.9603, 22.0317),
    'Hg': (4.92, -503083.5194,
           0.4942, 0.1002, 0.0000, 0.5336, 2.4200, 7.3056, 19.2470),
    'Tl': (5.11, -518127.6827,
           0.5246, 0.1342, 0.0000, 0.3435, 1.7529, 5.6330, 15.5894),
    'Pb': (5.26, -533425.7883,
           0.4718, 0.1195, 0.0000, 0.3368, 1.5778, 4.8543, 13.2736),
    'Bi': (5.24, -548977.7650,
           0.3959, 0.0444, 0.0000, 0.5429, 2.1727, 5.9091, 14.6560),
    'Po': (5.31, -564784.6694,
           0.3963, 0.0487, 0.0000, 0.5385, 2.0447, 5.2647, 12.9694),
    'At': (5.47, -580848.2407,
           0.3776, 0.0616, 0.0000, 0.4873, 1.8808, 4.3189, 10.1874),
    'Rn': (5.79, -597169.9258,
           0.2473, 0.0523, 0.0000, 0.2870, 1.2080, 3.0923, 6.8806),
    'Fr': (6.04, -613749.4809,
           0.1824, 0.0415, 0.0000, 0.2040, 0.9084, 2.5611, 6.0603),
    'Ra': (5.77, -630584.9307,
           0.4823, 0.1247, 0.0000, 0.3105, 1.3983, 3.8274, 8.7644),
    'Ac': (5.39, -647675.9322,
           0.8865, 0.2800, 0.0000, 0.3729, 1.9590, 5.4529, 12.2960),
    'Th': (5.08, -665028.2949,
           0.5337, 0.0560, 0.0000, 0.8066, 3.1140, 7.7431, 16.6047),
    'Pa': (4.94, -682646.0001,
           0.1771, -0.1418, 0.0000, 1.0409, 3.5954, nan, nan),
    'U':  (4.84, -700531.4637,
           -0.1523, -0.3182, 0.0000, 1.2219, 3.9585, 9.0891, 18.8877),
    'Np': (4.78, -718686.2289,
           -0.4553, -0.4672, 0.0000, 1.3403, 4.1582, 9.3422, nan),
    'Pu': (4.75, -737111.8551,
           -0.7052, -0.5909, 0.0000, 1.4267, 4.2593, 9.3982, nan),
    'Am': (4.74, -755809.8786,
           -0.8431, -0.6637, 0.0000, 1.4687, 4.2763, 9.3197, nan),
    'Cm': (4.74, -774781.7084,
           -0.8864, -0.6923, 0.0000, 1.5028, 4.2908, 9.2478, nan),
    'Bk': (4.77, -794028.8368,
           -0.8151, -0.6476, 0.0000, 1.4249, 4.1161, 8.8684, nan),
    'Cf': (4.80, -813552.4934,
           -0.7637, -0.6105, 0.0000, 1.3716, 3.9894, 8.6511, nan),
    'Es': (4.84, -833354.1488,
           -0.6847, -0.5525, 0.0000, 1.2898, 3.8101, 8.4030, nan),
    'Fm': (4.89, -853435.1305,
           -0.6257, -0.5024, 0.0000, 1.2390, 3.6934, nan, nan),
    'Md': (4.94, -873796.4863,
           -0.7057, -0.5467, 0.0000, 1.2703, 3.7328, nan, nan),
    'No': (5.02, -894437.5023,
           -0.1778, -0.2971, 0.0000, 1.0526, 3.5062, nan, nan)}
