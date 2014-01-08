import numpy as np
import ase.db
from ase.utils import prnt
from ase.data.g2_1 import molecule_names


c = ase.db.connect('g2-1.json')

def analyse(calc, relaxed):
    e = {}
    for d in c.select(natoms=1, calculator=calc):
        e[d.numbers[0]] = d.energy

    A = []
    for name in molecule_names:
        d = c.get(name=name, relaxed=relaxed, calculator=calc)
        ea = sum(e[Z] for Z in d.numbers) - d.energy
        dist = ((d.positions[0] - d.positions[-1])**2).sum()**0.5
        A.append((ea, dist))
    return np.array(A).T


En0, Dn0 = analyse('nwchem', 0)
En1, Dn1 = analyse('nwchem', 1)
Eg0, Dg0 = analyse('gpaw', 0)
Eg1, Dg1 = analyse('gpaw', 1)

print abs(Eg0 - En0).max(), abs(Eg0 - En0).mean()
print abs(Eg1 - En1).max(), abs(Eg1 - En1).mean()
print abs(Dg0 - Dn0).max(), abs(Dg0 - Dn0).mean()
print abs(Dg1 - Dn1).max(), abs(Dg1 - Dn1).mean()

fd = open('g2-1.csv', 'w')
prnt('# Atomization energies and distances.  N: NWChem, G: GPAW', file=fd)
prnt('# name, E(N, not relaxed), E(N), E(G)-E(N), d(N), d(G)-d(N)', file=fd)
for name, en0, en1, eg1, dn1, dg1 in zip(molecule_names,
                                         En0, En1, Eg1, Dn1, Dg1):
    prnt('%11s, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f' %
         (name, en0, en1, eg1 - en1, dn1, dg1 - dn1), file=fd)
    

# these results are calculated at relaxed geometries
ref = {'distance': {'NH3': 1.0212028325017757, 'S2': 1.909226444930773, 'SiH2_s3B1d': 1.4959758117701643, 'CH3OH': 1.103031397032791, 'SiH4': 1.492599477296688, 'Si2H6': 3.1894770187147787, 'PH3': 1.4301222209112225, 'PH2': 1.4346717268375369, 'HF': 0.93024767766719751, 'O2': 1.2177518292845226, 'SiH3': 1.4944876788686847, 'NH': 1.0495566124276352, 'SH2': 1.3513837051001727, 'ClO': 1.5787435668567111, 'H2O2': 1.8958330716525236, 'NO': 1.156790619328925, 'ClF': 1.6500936428996131, 'LiH': 1.6056281159928836, 'HCO': 1.134630847589132, 'CH3': 1.0857934634048991, 'CH4': 1.0954690924956099, 'Cl2': 2.0044430187644329, 'HOCl': 1.7084164068894601, 'SiH2_s1A1d': 1.5369100463950582, 'SiO': 1.5267775155499548, 'F2': 1.4131037387967056, 'P2': 1.9038007151393095, 'Si2': 2.2844499935377716, 'CH': 1.135943649633214, 'CO': 1.1353228578574415, 'CN': 1.1734204501040257, 'LiF': 1.5781425105489659, 'Na2': 3.0849004417809258, 'SO2': 1.4536010904565573, 'NaCl': 2.375956045377166, 'Li2': 2.7269446448266184, 'NH2': 1.0360313462587714, 'CS': 1.5453598419904586, 'C2H6': 2.1839921464514007, 'N2': 1.1020010395108386, 'C2H4': 2.1196102605939493, 'HCN': 1.074902699795437, 'C2H2': 1.0701025663671189, 'CH2_s3B1d': 1.0846126602770281, 'CH3Cl': 1.0934062531916822, 'BeH': 1.3549994786114516, 'CO2': 1.1705025664593323, 'CH3SH': 1.0951914666413829, 'OH': 0.98302300558682143, 'N2H4': 1.9949378750277829, 'H2O': 0.9689957468077689, 'SO': 1.5015729306543228, 'CH2_s1A1d': 1.1216809054426011, 'H2CO': 2.0340405586023551, 'HCl': 1.2882940110553078}, 'energy': {'NH3': -1537.8943922528292, 'S2': -21662.654140653412, 'SiH2_s3B1d': -7903.3690892881705, 'Li': -203.05091256761085, 'CH3OH': -3146.7647944133632, 'SiH4': -7938.4648222008409, 'Si2H6': -15845.082177930506, 'PH3': -9333.4084018049416, 'PH2': -9316.1298787915111, 'HF': -2732.0742553639361, 'O2': -4088.7283614591165, 'SiH3': -7920.9111544133084, 'NH': -1501.4257523508722, 'Be': -398.09853859103947, 'SH2': -10863.938964816221, 'ClO': -14561.296690949923, 'H2O2': -4121.946252450156, 'NO': -3532.6917094247037, 'ClF': -15231.961288082202, 'LiH': -218.97262833348015, 'HCO': -3096.2005211539013, 'CH3': -1082.8017329696315, 'CH4': -1101.1776994482279, 'Cl2': -25035.886547435726, 'Na': -4412.8227996808337, 'HOCl': -14578.981322327132, 'SiH2_s1A1d': -7904.0726269700554, 'SiO': -9920.2118092650708, 'F2': -5426.9009851655201, 'P2': -18569.7100395737, 'Si2': -15744.418687547124, 'C': -1028.5471962651873, 'CH': -1045.8247164754753, 'CO': -3081.4550529867706, 'CN': -2521.0913000695346, 'F': -2712.30655836285, 'H': -13.60405569717515, 'LiF': -2921.3671571474624, 'O': -2041.2483769244884, 'N': -1483.981485107872, 'Na2': -8826.410126700297, 'P': -9282.2144876505899, 'Si': -7870.4445140229245, 'SO2': -14923.501322516569, 'NaCl': -16933.432926277521, 'Li2': -406.96985635618387, 'NH2': -1519.3756700462491, 'CS': -11865.161113573042, 'C2H6': -2169.7998017018276, 'N2': -2978.5280844918202, 'C2H4': -2136.2951883618102, 'HCN': -2540.2803982015039, 'C2H2': -2102.2901648364796, 'CH2_s3B1d': -1064.1920578821023, 'CH3Cl': -13603.223029214254, 'BeH': -414.1082135671578, 'CO2': -5129.0868582477142, 'CH3SH': -11932.535768470407, 'OH': -2059.6213922996608, 'Cl': -12516.517962625088, 'S': -10828.826656949115, 'N2H4': -3042.0346586048404, 'H2O': -2078.6212796010145, 'SO': -12876.201520592005, 'CH2_s1A1d': -1063.5161597336582, 'H2CO': -3113.7397623037459, 'HCl': -12534.742429020402}, 'ea': {'NH3': 13.100740053431537, 'S2': 5.0008267551820609, 'SiH2_s3B1d': 5.7164638708964048, 'CH3OH': 22.552998434986876, 'SiH4': 13.604085389217289, 'Si2H6': 22.568815701608401, 'PH3': 10.381747062827344, 'PH2': 6.7072797465716576, 'HF': 6.1636413039109357, 'O2': 6.2316076101396902, 'SiH3': 9.654473298859557, 'NH': 3.8402115458250137, 'SH2': 7.9041964727566665, 'ClO': 3.5303514003462624, 'H2O2': 12.241387206829131, 'NO': 7.4618473923433157, 'ClF': 3.1367670942636323, 'LiH': 2.3176600686941526, 'HCO': 12.800892267050585, 'CH3': 13.442369612918583, 'CH4': 18.214280394339767, 'Cl2': 2.8506221855495824, 'HOCl': 7.6109270803808613, 'SiH2_s1A1d': 6.4200015527812866, 'SiO': 8.5189183176571532, 'F2': 2.2878684398201585, 'P2': 5.2810642725198704, 'Si2': 3.5296595012750913, 'CH': 3.6734645131127763, 'CO': 11.659479797095173, 'CN': 8.5626186964755107, 'LiF': 6.009686217001672, 'Na2': 0.7645273386297049, 'SO2': 12.177911718477844, 'NaCl': 4.0921639715998026, 'Li2': 0.86803122096216612, 'NH2': 8.1860735440266126, 'CS': 7.7872603587384219, 'C2H6': 31.081074988401724, 'N2': 10.565114276076201, 'C2H4': 24.784573042734792, 'HCN': 14.147661131269615, 'C2H2': 17.987660911754574, 'CH2_s3B1d': 8.4367502225645694, 'CH3Cl': 17.345703232453161, 'BeH': 2.4056192789431634, 'CO2': 18.042908133550554, 'CH3SH': 20.745692467404297, 'OH': 4.7689596779973726, 'N2H4': 19.655465600395473, 'H2O': 10.164791282175884, 'SO': 6.126486718401793, 'CH2_s1A1d': 7.7608520741205211, 'H2CO': 16.736077719720015, 'HCl': 4.6204106981385848}}
En10 = [ref['ea'][name] for name in molecule_names]
Dn10 = [ref['distance'][name] for name in molecule_names]
print abs(En1 - En10).max()
print abs(Dn1 - Dn10).max()


