import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.test import only_on_master
from gpaw.mpi import world
from . import check_txt_data
from ase.parallel import paropen
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.qed import RRemission
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

@pytest.fixture()
@only_on_master(world)
def write_inputs():
    for label in 'xyz':
        with paropen(f'dm_ensemblebare_{label}.dat', 'w') as fd:
            fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000      -4.92017275e-16     2.055079955338e-14     2.275267093410e-14     9.816676572374e-15
# Kick = [    0.000000000000e+00,     0.000000000000e+00,     1.000000000000e-05]; Time = 0.00000000
          0.00000000      -2.56261929e-15     5.578074164489e-14     4.477138474129e-14     4.724849004460e-14
          0.41341373       3.79876430e-16    -1.137633546705e-14    -2.752339225899e-14     9.139522195897e-06
          0.82682747       1.82844350e-15    -3.229411358388e-14    -2.899130651281e-14     1.822151056685e-05
          1.24024120       2.30257796e-16     2.568849944173e-15    -1.467914253813e-15     2.719465212110e-05
          1.65365493      -1.09563085e-16     1.834892817266e-14    -3.669785634532e-16     3.601753782894e-05
          2.06706867       1.33270742e-15    -2.422058518791e-14    -3.045922076662e-14     4.466069760315e-05
          2.48048240       8.37044574e-16    -1.724799248230e-14    -2.642245656863e-14     5.310709155553e-05
          2.89389613       2.93431494e-17     5.504678451798e-15     2.568849944173e-15     6.135047097171e-05
          3.30730987       3.77197226e-16     1.100935690360e-15    -0.000000000000e+00     6.939195199957e-05
          3.72072360      -1.63710169e-16     1.431216397468e-14     6.238635578705e-15     7.723547141815e-05
          4.13413733      -4.03410565e-16     2.055079955338e-14     4.403742761439e-15     8.488299714725e-05
          4.54755107      -2.41349366e-15     6.752405567539e-14     7.596456263482e-14     9.233040644308e-05
          4.96096480       1.00069275e-15    -2.568849944173e-14    -3.449598496460e-14     9.956481323431e-05
          5.37437853      -2.31624289e-15     6.642311998503e-14     6.312031291396e-14     1.065638527118e-04
          5.78779227      -3.55273735e-15     8.697391953842e-14     8.293715534043e-14     1.132970565629e-04
          6.20120600       1.69532988e-15    -3.192713502043e-14    -3.229411358388e-14     1.197290637407e-04
          6.61461974      -2.82658143e-15     6.091844153324e-14     7.706549832518e-14     1.258240590975e-04
          7.02803347       2.57310750e-15    -4.697325612201e-14    -4.954210606619e-14     1.315505996358e-04
          7.44144720      -9.51935488e-17     7.339571269065e-16     1.137633546705e-14     1.368859163511e-04
          7.85486094       2.68573513e-15    -5.174397744691e-14    -4.844117037583e-14     1.418188643634e-04
          8.26827467       1.00204830e-15    -2.275267093410e-14    -2.568849944173e-15     1.463509522156e-04
          8.68168840       2.04459745e-15    -4.660627755856e-14    -3.743181347223e-14     1.504952279150e-04
          9.09510214      -4.46448289e-16     1.064237834014e-14     1.871590673611e-14     1.542732031233e-04
          9.50851587      -2.70445314e-15     6.789103423885e-14     6.752405567539e-14     1.577103565344e-04
          9.92192960       2.74605367e-15    -5.798261302561e-14    -6.679009854849e-14     1.608310263102e-04
    '''.strip())  # noqa: E501


def test_embeddingrr_dyadic(in_tmp_dir, write_inputs):
    """
    Testing if the dyadic is calculated correctly
    """
    dt, nt = 0.1, 1000
    rremission = RRemission(1e1, [0, 0, 1],
                            [2.98, 3, 0.01, 10, 5000, 3000], [1, 1, 0, 0, 0])
    dyadic_out_very_good = rremission.dyadicGt(dt, nt)
    # Test that the first 30 elements of the dyadic agree with the reference
    dyadic_ref = [-0.26509844 + 1.07477307e-15j, -0.2646285 + 5.76719985e-16j,
                  -0.26482334 + 2.78241075e-16j, -0.26470851 - 2.48979644e-16j,
                  -0.26446162 - 4.63244966e-16j, -0.26435981 + 5.00200484e-16j,
                  -0.26407471 + 1.93115042e-16j, -0.26387885 - 1.57573539e-16j,
                  -0.26355268 + 7.27542009e-16j, -0.26327364 + 4.10345111e-16j,
                  -0.26289701 + 2.23836927e-16j, -0.26254291 - 1.52533693e-16j,
                  -0.26211057 - 4.12098469e-16j, -0.26168581 - 2.42895168e-16j,
                  -0.26119494 - 4.52349475e-16j, -0.26070207 + 1.01101853e-16j,
                  -0.26015115 + 4.87958740e-16j, -0.25959181 + 3.62597864e-16j,
                  -0.25898007 - 4.87213351e-16j, -0.25835544 + 2.55465137e-16j,
                  -0.25768254 + 5.68664039e-16j, -0.25699357 - 4.40321607e-16j,
                  -0.25625944 - 2.43674438e-16j, -0.2555069 + 8.38901431e-16j,
                  -0.25471166 + 5.84927072e-16j, -0.25389627 - 5.16080234e-16j,
                  -0.25304015 - 7.72222997e-16j, -0.25216259 - 3.98986399e-16j,
                  -0.25124595 + 4.53738609e-16j, -0.25030692 + 4.82469967e-16j]
    assert np.allclose(dyadic_out_very_good[:30], dyadic_ref)


def test_embeddingrr(in_tmp_dir, write_inputs):
    """
    Perform a short TD run to check if the full function remained consistent
    """
    d = 1.104  # N2 bondlength
    atoms = Atoms('Na2',
                  positions=[[0, 0, d / 2], [0, 0, -d / 2]],
                  cell=[10, 10, 10])
    atoms.center()
    calc = GPAW(mode='lcao', h=0.4, basis='dzp',
                setups={'Na': '1'},
                convergence={'density': 1e-12})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')
    td_calc = LCAOTDDFT('gs.gpw',
                        rremission=RRemission(1e1, [0, 0, 1],
                                              [2.98, 3, 0.01, 10, 5000, 3000],
                                              [1, 1, 0, 0, 0]))
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(1, 50)

    with paropen('dm_ref.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000       7.88604804e-17    -4.619182583622e-15    -1.894605815478e-15    -1.210967459879e-14
# Kick = [    0.000000000000e+00,     0.000000000000e+00,     1.000000000000e-05]; Time = 0.00000000
          0.00000000       2.34993646e-17     2.794421682618e-15    -6.928648854704e-16     1.025270002303e-14
          0.04134137       5.26406938e-16    -6.060129843631e-15    -5.745327645577e-15     9.485684350114e-07
          0.08268275       1.22403109e-15    -7.852593722727e-15    -8.850217474278e-15     1.897048614147e-06
          0.12402412       6.24347421e-16    -6.988742152240e-15    -9.706984536733e-15     2.845351886780e-06
          0.16536549      -2.89507938e-17     5.599553474437e-15     3.564507705631e-15     3.793389615319e-06
          0.20670687       8.40208139e-17    -3.142187679807e-16    -7.821296866659e-16     4.741073368372e-06
          0.24804824       5.55437122e-16    -2.278127741436e-15    -7.309962081099e-15     5.688314948712e-06
          0.28938961      -5.09682165e-17     8.917311932693e-16    -2.643354967243e-15     6.635026534065e-06
          0.33073099       7.32635034e-16    -3.123809632502e-15    -8.615261849095e-15     7.581120643251e-06
          0.37207236       5.52367385e-16    -7.976655960430e-15    -6.614263392416e-15     8.526510288283e-06
          0.41341373       4.02347230e-16    -3.959656559476e-15    -3.944279009690e-15     9.471108989504e-06
          0.45475511      -1.62378493e-16    -1.895522634164e-15    -2.525752133922e-15     1.041483085015e-05
          0.49609648       5.36489436e-16    -6.912396159808e-15    -1.340997353759e-14     1.135759060280e-05
          0.53743785       9.72524381e-17     2.724785136028e-15     1.145523275100e-15     1.229930372672e-05
          0.57877923       4.07163541e-16    -4.735118473877e-15    -3.472409101223e-15     1.323988643470e-05
          0.62012060       1.17761456e-17    -4.131310021719e-15    -1.461408986147e-15     1.417925580091e-05
          0.66146197       1.44089742e-15    -1.396685754243e-14    -1.594139327794e-14     1.511732973992e-05
          0.70280335       9.15337301e-16    -8.072713554620e-15    -3.976534358022e-15     1.605402719110e-05
          0.74414472      -1.74101712e-16    -6.889642386954e-15    -5.253204378855e-15     1.698926805586e-05
          0.78548609       1.38926762e-15    -1.026961949516e-14    -1.284192101420e-14     1.792297327241e-05
          0.82682747       8.40181675e-16    -1.092814535346e-14    -1.148544609407e-14     1.885506497411e-05
          0.86816884       1.65130670e-17    -5.435401255992e-15    -1.933820651111e-15     1.978546641072e-05
          0.90951021       5.94946752e-16    -6.384808679352e-16     3.100347408845e-15     2.071410206988e-05
          0.95085159       1.18370110e-16    -3.421358969820e-15    -2.631728039356e-15     2.164089773755e-05
          0.99219296      -1.81405568e-16    -2.606348831173e-15    -1.392480890359e-15     2.256578054454e-05
          1.03353433       1.17613261e-15    -8.299792873815e-15    -8.519329275635e-15     2.348867893594e-05
          1.07487571       8.11098565e-17    -1.589013477865e-15    -1.682403963149e-15     2.440952290225e-05
          1.11621708       7.06886293e-16    -3.868516447331e-15     5.881391873357e-16     2.532824380245e-05
          1.15755845       1.91567456e-16    -4.472783308832e-15    -6.433733458346e-15     2.624477458254e-05
          1.19889983       1.34486229e-15    -9.041290823242e-15    -8.750534278918e-15     2.715904972135e-05
          1.24024120      -5.51097149e-16     4.863306395669e-17     1.849598352690e-15     2.807100536919e-05
          1.28158257       9.34681936e-17    -2.702614793247e-15    -2.522918330709e-15     2.898057922810e-05
          1.32292395      -7.67434205e-18    -5.652437243213e-15    -4.251871678983e-15     2.988771077659e-05
          1.36426532       6.91378829e-16    -7.550168576940e-15    -7.514787710360e-15     3.079234118381e-05
          1.40560669       6.12491886e-16    -1.083467152102e-14    -9.592923957428e-15     3.169441340928e-05
          1.44694807       1.06665416e-15    -7.773205559199e-15    -7.890349983177e-15     3.259387218059e-05
          1.48828944       4.13197161e-16    -5.305254676098e-15    -7.145726515501e-15     3.349066407034e-05
          1.52963081       1.24533401e-15    -1.609275170835e-14    -1.358233545064e-14     3.438473749645e-05
          1.57097219       8.00407413e-16    -9.214486207821e-15    -1.138034533375e-14     3.527604281661e-05
          1.61231356       2.87285025e-16    -3.200530687124e-15    -3.789253304533e-15     3.616453224279e-05
          1.65365493       5.77163449e-17    -2.610599535992e-15     5.614264246997e-16     3.705015996532e-05
          1.69499631       6.57611724e-16    -7.851718577618e-15    -6.863388033661e-15     3.793288211549e-05
          1.73633768       5.91400677e-16    -7.974280566561e-15    -3.723409053418e-15     3.881265684789e-05
          1.77767905      -2.69104774e-16     3.298296897957e-15     2.471826525730e-15     3.968944428790e-05
          1.81902043      -4.81075394e-16    -2.471243095657e-17     1.306841690332e-15     4.056320658627e-05
          1.86036180       1.70158688e-17    -1.493164251558e-15    -2.349181189634e-15     4.143390795873e-05
          1.90170317       7.76220004e-16    -6.716155287338e-15    -2.052548671002e-15     4.230151462909e-05
          1.94304455      -3.04168578e-16     4.050671650891e-16     4.508997646946e-15     4.316599493963e-05
          1.98438592       4.68081939e-16    -4.798879046160e-15    -1.174278042992e-15     4.402731922395e-05
          2.02572729       9.60721772e-16    -6.887100298778e-15    -7.528665011386e-15     4.488545996108e-05
          2.06706867      -2.77043748e-16     6.691942939302e-16     1.013293016373e-15     4.574039168674e-05
'''.strip())  # noqa: E501

    check_txt_data('dm.dat', 'dm_ref.dat', atol=1e-8)
