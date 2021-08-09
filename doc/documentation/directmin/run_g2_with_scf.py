from ase.collections import g2
from gpaw import GPAW, LCAO, FermiDirac, ConvergenceError
from ase.parallel import parprint
from gpaw.utilities.memory import maxrss
import time
from gpaw.mpi import world

xc = 'PBE'
mode = LCAO()

file2write = open('scf-g2-results.txt', 'w')

for name in g2.names:

    atoms = g2[name]
    if len(atoms) == 1:
        continue
    atoms.center(vacuum=7.0)
    calc = GPAW(xc=xc, h=0.15,
                convergence={'density': 1.0e-6},
                maxiter=333,
                basis='dzp',
                mode=mode,
                txt=name + '.txt',
                occupations=FermiDirac(width=0.0, fixmagmom=True),
                symmetry='off'
                )

    atoms.calc = calc

    try:
        t1 = time.time()
        e = atoms.get_potential_energy()
        t2 = time.time()
        steps = atoms.calc.get_number_of_iterations()
        memory = maxrss() / 1024.0 ** 2
        parprint(name +
                 "\t{}\t{}\t{}\t{:.3f}".format(steps, e,
                                               t2-t1, memory), file=file2write, flush=True)  # s,MB
    except ConvergenceError:
        parprint(name +
                 "\t{}\t{}\t{}\t{}".format(None, None, None, None), file=file2write, flush=True)
    calc = None
    atoms = None

file2write.close()

output = \
"""
PH3	18	-15.067939739162565	9.424316883087158	345.191
P2	14	-8.379364578606403	6.812095403671265	345.191
CH3CHO	17	-37.752640153790544	12.035242795944214	398.117
H2COH	18	-23.700059799196794	15.379578351974487	463.512
CS	15	-9.417393294333172	6.916038513183594	463.512
OCHCHO	17	-35.66811875892615	10.182486534118652	463.512
C3H9C	19	-66.59205808003622	23.414774656295776	566.285
CH3COF	17	-38.4908567580481	10.962944507598877	566.285
CH3CH2OCH3	17	-61.30320737242064	14.620186567306519	566.285
HCOOH	18	-28.40857295188662	10.705278158187866	566.285
HCCl3	16	-17.403156748455167	11.3733069896698	566.285
HOCl	16	-9.765718189549954	8.294231176376343	566.285
H2	12	-6.490910413647257	5.156960725784302	566.285
SH2	18	-10.506376963353159	8.987826108932495	566.285
C2H2	15	-21.8315594881508	7.5802977085113525	566.285
C4H4NH	17	-60.36737073551913	12.462259769439697	566.285
CH3SCH3	18	-42.75755391907749	13.17004919052124	566.285
SiH2_s3B1d	15	-8.439652470852263	13.359965801239014	566.285
CH3SH	19	-26.57600571342069	12.460690259933472	566.285
CH3CO	15	-32.727526205821974	15.398682832717896	566.285
CO	14	-13.817286401313696	6.160528659820557	566.285
ClF3	15	-7.280418158543331	8.876623153686523	566.285
SiH4	16	-18.406906983682326	9.468269348144531	566.285
C2H6CHOH	16	-61.727709924460434	12.179592609405518	566.285
CH2NHCH2	17	-42.170691521207104	11.65274715423584	566.285
isobutene	17	-63.67196360060958	14.193466663360596	566.285
HCO	17	-16.12661908530326	14.049065589904785	566.285
bicyclobutane	15	-54.50318222862183	12.325263738632202	566.285
LiF	18	-5.818530975164816	7.934077739715576	566.285
C2H6	18	-39.52761899945607	11.584608554840088	566.285
CN	17	-12.013574497449207	12.21487021446228	566.285
ClNO	18	-13.637173578862363	9.969871997833252	566.285
SiF4	15	-25.21531130705909	9.05084753036499	566.285
H3CNH2	15	-34.542907090515556	9.589577198028564	566.285
methylenecyclopropane	17	-54.687880878011434	12.64513349533081	566.285
CH3CH2OH	15	-45.38267490450617	9.680669069290161	566.285
NaCl	16	-3.991481648956574	7.777966499328613	566.285
CH3Cl	18	-21.364671136168802	11.272502183914185	566.285
CH3SiH3	18	-34.957867383396355	12.658702850341797	566.285
AlF3	15	-18.70969513486594	7.844922065734863	566.285
C2H3	20	-24.912867583547744	18.332961082458496	566.285
ClF	15	-3.0836445438877313	6.874228239059448	566.285
PF3	16	-17.620133982392183	10.052327394485474	566.285
PH2	20	-10.333156669751412	16.9914813041687	566.285
CH3CN	17	-35.399551652561215	11.100446939468384	566.285
cyclobutene	15	-55.09416925492654	10.81662106513977	566.285
CH3ONO	22	-37.484991962456576	12.836097240447998	566.285
SiH3	19	-13.414822942888721	18.2525372505188	566.285
C3H6_D3h	15	-47.164632312754364	9.059163093566895	566.285
CO2	14	-21.690591409424943	6.678245782852173	566.285
NO	163	-11.327965166239705	110.53826141357422	566.285
trans-butane	17	-71.89452926449687	14.369985818862915	566.285
H2CCHCl	19	-29.238622363016635	11.707709550857544	566.285
LiH	17	-3.5313140027080223	7.427795171737671	566.285
NH2	13	-12.76557469532906	10.299749374389648	566.285
CH	80	-5.673636310132295	54.34945034980774	566.285
CH2OCH2	18	-36.76395515833223	11.68323564529419	566.285
C6H6	15	-73.85339953604031	12.10660433769226	566.285
CH3CONH2	19	-50.168313481824775	13.152329921722412	566.285
cyclobutane	16	-63.539274905937184	13.763991832733154	566.285
H2CCHCN	18	-43.1621120373009	12.488061666488647	566.285
butadiene	19	-55.31058655984656	13.683945655822754	566.285
H2CO	15	-21.139006626605422	7.932563543319702	566.285
CH3COOH	19	-44.924174089136265	12.247894287109375	566.285
HCF3	16	-23.33279099725403	8.660435438156128	566.285
CH3S	17	-21.690728087340183	17.465802907943726	566.285
CS2	17	-15.408150740029571	8.645901679992676	566.285
SiH2_s1A1d	19	-9.141474384102668	9.489758491516113	566.285
C4H4S	19	-52.339132780598575	13.792178630828857	566.285
N2H4	15	-29.04650645552515	8.978219032287598	566.285
OH	156	-6.984028868322686	105.05806231498718	566.285
CH3OCH3	16	-44.99641117875515	11.272940158843994	566.285
C5H5N	19	-69.10427719659162	14.708759069442749	566.285
H2O	16	-13.250986473888151	7.394877195358276	566.285
HCl	16	-5.36259443336188	6.883431434631348	566.285
CH2_s1A1d	17	-10.75750241996795	7.841189384460449	566.285
CH3CH2SH	18	-42.77177811221571	11.519938468933105	566.285
CH3NO2	17	-37.59617685073973	11.368172407150269	566.285
BCl3	15	-14.567528269012527	9.441259145736694	566.285
C4H4O	18	-54.63443308676421	12.29889440536499	566.285
CH3O	17	-23.46123621228536	16.68055820465088	566.285
CH3OH	15	-29.061696621373784	9.963403224945068	566.285
C3H7Cl	18	-53.83150398571328	12.202645301818848	566.285
isobutane	18	-71.94286533151197	14.485913515090942	566.285
CCl4	15	-15.275216282079997	10.258109092712402	566.285
CH3CH2O	17	-39.64358031935646	17.93339967727661	592.074
H2CCHF	16	-30.90462662415892	8.886056423187256	592.074
C3H7	19	-50.22810186101624	23.692474126815796	611.449
CH3	15	-17.49423441816605	12.92069125175476	611.449
O3	17	-12.436218920717389	8.096709251403809	611.449
C2H4	15	-30.960041413667845	8.145560503005981	611.449
NCCN	17	-30.656829904791742	8.859314441680908	611.449
S2	17	-6.219212745370212	13.239931583404541	611.449
AlCl3	15	-12.985497496691243	10.23410153388977	611.449
SiCl4	16	-16.784060452785397	10.895533800125122	611.449
SiO	17	-10.264395373463588	7.712653160095215	611.449
C3H4_D2d	16	-38.567667331353746	11.068986892700195	611.449
COF2	16	-22.143807001114947	9.37898874282837	611.449
2-butyne	19	-54.940138451098214	14.225937128067017	611.449
C2H5	17	-33.851029358002215	17.2315890789032	611.449
BF3	14	-21.57253092881569	8.061803579330444	611.449
N2O	19	-20.101453880304287	8.767191648483276	611.449
F2O	15	-7.401790642950211	7.697839736938477	611.449
SO2	15	-14.870076420824926	8.16426944732666	611.449
H2CCl2	19	-19.399891835301172	12.314615726470947	611.449
CF3CN	19	-34.90747376810633	13.380947351455688	611.449
HCN	17	-18.729247544127606	8.031611919403076	611.449
C2H6NH	15	-50.52638663455311	11.172731399536133	611.449
OCS	17	-18.602824279134886	8.478620529174805	611.449
ClO	117	-4.694492551841379	83.39384412765503	611.449
C3H8	18	-55.71715479135624	13.572690486907959	611.449
HF	15	-7.004240357122936	6.443803548812866	611.449
O2	15	-9.111654239498307	10.66483449935913	611.449
SO	35	-7.921946288251748	25.569963932037354	611.449
NH	16	-7.488300012131835	11.436579465866089	611.449
C2F4	14	-30.241856053715637	8.400748491287231	611.449
NF3	15	-14.066501151063182	7.418868780136108	611.449
CH2_s3B1d	16	-11.46398769462341	13.001952409744263	611.449
CH3CH2Cl	18	-37.64667833838129	10.695473909378052	611.449
CH3COCl	17	-36.58066815772619	12.28981900215149	611.449
NH3	13	-18.686996789673476	7.214409112930298	611.449
C3H9N	16	-66.5765684297463	11.513446569442749	611.449
CF4	17	-23.614422687205757	10.08606219291687	611.449
C3H6_Cs	18	-47.31966084193544	11.612122058868408	611.449
Si2H6	15	-30.154437351754027	11.447124481201172	611.449
HCOOCH3	19	-44.26973461011855	12.293081283569336	611.449
CCH	16	-14.799409011958037	12.531543731689453	611.449
Si2	17	-4.793115114511537	13.495885133743286	611.449
C2H6SO	18	-48.02735877864575	13.002910852432251	611.449
C5H8	15	-70.91659810523136	12.42467713356018	611.449
H2CF2	16	-22.971190008076956	9.946171045303345	611.449
Li2	21	-1.3181781742775984	9.724149942398071	611.449
CH2SCH2	20	-34.71520519895414	13.514520168304443	611.449
C2Cl4	17	-23.595139969720563	10.92472243309021	611.449
C3H4_C3v	16	-38.416472047250096	11.086548089981079	611.449
CH3COCH3	17	-54.26678298560979	12.976754903793335	611.449
F2	16	-2.7789800209690525	7.044575214385986	611.449
CH4	15	-23.40050852386761	7.613523483276367	611.449
SH	35	-5.401240463979583	25.431840181350708	611.449
H2CCO	15	-29.57475371469119	8.67544960975647	611.449
CH3CH2NH2	16	-50.82201231180034	10.044011116027832	611.449
N2	18	-15.60692193925049	7.576152324676514	611.449
Cl2	15	-2.535145846065773	6.989879846572876	611.449
H2O2	17	-16.958825223285647	9.133270025253296	611.449
Na2	11	-1.0643211358816327	5.979414701461792	611.449
BeH	45	-2.9158429067846203	31.71684956550598	611.449
C3H4_C2v	16	-37.684405607781144	11.153424739837646	611.449
NO2	16	-17.167930472676215	13.106889963150024	611.449
"""

import numpy as np
output.splitlines()

# this is saved data
saved_data = {}
for i in output.splitlines():
    if i == '':
        continue
    mol = i.split()
    # ignore last two columns which are memory and elapsed time
    saved_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])

file2read = open('scf-g2-results.txt', 'r')
calculated_data_string = file2read.read().split('\n')
file2read.close()

# this is data calculated, we would like to coompare it to saved
# compare number of iteration, energy and gradient evaluation,
# and energy

calculated_data = {}
for i in calculated_data_string:
    if i == '':
        continue
    mol = i.split()
    # ignore last two columns which are memory and elapsed time
    calculated_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])

error = np.array([3, 3, 1.0e-3])

assert len(calculated_data) == len(saved_data)
for k in saved_data.keys():
    assert (abs(saved_data[k] - calculated_data[k]) < error).all()

