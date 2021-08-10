from ase import Atoms
from gpaw import GPAW
atoms = Atoms('(O2Ti4O6)3V2O5',
              [(8.3411, 1.9309, 9.2647),
               (3.2338, 0.0854, 9.4461),
               (1.5783, 0.1417, 8.4327),
               (6.6126, 2.0588, 8.2126),
               (9.4072, 2.0891, 7.6857),
               (4.2748, 0.1256, 7.9470),
               (9.9477, 0.2283, 7.3281),
               (4.7391, 2.0618, 7.6111),
               (7.7533, 2.1354, 6.7529),
               (2.6347, 0.2585, 6.9403),
               (6.2080, 0.1462, 8.6280),
               (0.9517, 1.9798, 8.9832),
               (8.5835, 6.2280, 9.4350),
               (3.1930, 3.9040, 9.7886),
               (1.4899, 4.0967, 8.8898),
               (6.6167, 5.8865, 8.8463),
               (9.3207, 5.9258, 7.7482),
               (4.1984, 3.9386, 8.2442),
               (9.9075, 3.9778, 7.6337),
               (4.7626, 5.8322, 8.0051),
               (7.6143, 5.6963, 7.1276),
               (2.5760, 3.9661, 7.3115),
               (6.2303, 3.8223, 8.8067),
               (1.1298, 5.9913, 8.6968),
               (8.3845, 9.7338, 9.1214),
               (3.1730, 7.9593, 9.3632),
               (1.5914, 7.8120, 8.2310),
               (6.7003, 9.7064, 8.1528),
               (9.3943, 9.7202, 7.6037),
               (4.3168, 7.7857, 7.9666),
               (9.9045, 7.7968, 7.2716),
               (4.7772, 9.7015, 7.4648),
               (7.7314, 9.7221, 6.6253),
               (2.7673, 7.6929, 6.8222),
               (6.2358, 7.8628, 8.6557),
               (1.0528, 9.7017, 8.5919),
               (8.4820, 5.0952, 11.4981),
               (9.7787, 2.0447, 11.0800),
               (6.4427, 5.6315, 10.7415),
               (10.7389, 0.4065, 11.8697),
               (8.3109, 3.0048, 12.4083),
               (10.4702, 4.1612, 10.6543),
               (8.9827, 6.3884, 13.0109)],
              cell=[10.152054, 11.430000, 18.295483],
              pbc=[1, 1, 0])

atoms.calc = GPAW(h=0.20,
                  kpts=(2, 2, 1),
                  xc='RPBE')
ncpus = 8
