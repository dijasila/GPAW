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
    Testing if the dyadic is calculated correctly.
    Note, that this requires to use "extrapolate" for the interp1d function
    """
    dt, nt = 0.1, 1000
    rremission = RRemission(1e1, [0, 0, 1],
                            [2.98, 3, 0.01, 10, 5000, 3000], [1, 1, 0, 0, 0])
    dyadic_out_very_good, _ = rremission.dyadicGt(dt, nt)
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
    assert np.allclose(dyadic_out_very_good[:30, -1], dyadic_ref)

    """
    Test if doubling resolution = half total time for dyadic
    """
    dt, nt = 0.1, 500
    rremission = RRemission(1e1, [0, 0, 1],
                            [2.98, 3, 0.01, 10, 5000, 6000], [1, 1, 0, 0, 0])
    dyadic_out_very_good_split, _ = rremission.dyadicGt(dt, nt)
    assert np.allclose(dyadic_out_very_good_split[:30, -1], dyadic_ref)


def test_embeddingrr(in_tmp_dir, write_inputs):
    cross_section = 1e-5
    dt, nt = 5, 100
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
                        rremission=RRemission(1e-5, [0, 0, 1],
                                              [2.98, 3, 0.01, 10, 5000, 30000],
                                              [1, 1, 0, 0, 0]))
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(dt, nt)

    with paropen('dm_ref.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000       1.44357020e-16    -1.055049940181e-15    -1.183029494090e-15    -2.047822887409e-14
# Kick = [    0.000000000000e+00,     0.000000000000e+00,     1.000000000000e-05]; Time = 0.00000000
          0.00000000      -4.80519666e-16     2.313133545827e-15    -3.058840526496e-16    -2.173477055740e-14
          0.20670687       1.02219590e-15    -7.356511466223e-15    -4.051421775271e-15     4.732601922742e-06
          0.41341373       9.31770978e-17    -1.656399651317e-15     5.082926144642e-16     9.446720751073e-06
          0.62012060       1.18661206e-16    -3.635477806675e-15    -3.760790251677e-15     1.416967714122e-05
          0.82682747      -2.67411126e-16     4.101013331491e-15     2.141230042124e-15     1.894449631200e-05
          1.03353433      -3.43492965e-17    -1.214784759501e-16     1.526211397846e-15     2.379196983014e-05
          1.24024120       3.14171686e-16    -5.132309332979e-15    -5.322715904716e-15     2.870930972984e-05
          1.44694807       5.59618315e-16    -2.329844650066e-15    -4.913564729115e-15     3.368508520470e-05
          1.65365493       6.18922455e-16    -3.419275290987e-15    -5.683984140741e-15     3.870826566340e-05
          1.86036180      -2.74609129e-16     3.960865093199e-15     3.056340111897e-15     4.376871795247e-05
          2.06706867      -1.95484016e-16    -5.778874874785e-16     2.177027644471e-16     4.885706136421e-05
          2.27377553       3.41031883e-16     1.899314929640e-15    -1.877061239706e-15     5.396474866038e-05
          2.48048240      -8.05012018e-16     8.284915406949e-15     7.957819503782e-15     5.908416789129e-05
          2.68718927      -8.00407413e-16     8.465028605253e-15     1.153587112183e-14     6.420877924589e-05
          2.89389613       6.67905928e-16    -6.262580079022e-15    -7.607886480608e-15     6.933318707366e-05
          3.10060300       2.12949760e-16    -5.181400806278e-15    -3.968908093494e-15     7.445315578400e-05
          3.30730987       5.91691772e-16    -4.774333309510e-15    -4.908188837727e-15     7.956558925325e-05
          3.51401673       9.26001990e-16    -5.995910862005e-15    -6.719822562084e-15     8.466846332162e-05
          3.72072360       7.44067157e-16    -3.548046642852e-15    -5.911938605045e-15     8.976071062094e-05
          3.92743047       7.42188266e-16    -9.658268125623e-16    -4.221908377368e-15     9.484205954609e-05
          4.13413733       3.09223058e-16     1.139355585755e-15    -3.333886132421e-19     9.991283056580e-05
          4.34084420      -5.19791126e-16     2.293713659106e-15     1.252332652067e-15     1.049736912047e-04
          4.54755107      -6.74495277e-16     4.908813941376e-15     7.692400494064e-15     1.100253732341e-04
          4.75425793       1.47821060e-15    -9.570170184574e-15    -1.540897166259e-14     1.150683589387e-04
          4.96096480       2.55714370e-16    -2.990704228665e-15    -2.852056239133e-15     1.201025501765e-04
          5.16767167       7.10061883e-16    -9.629221642694e-15    -9.957109343818e-15     1.251269404338e-04
          5.37437853      -2.52035978e-16     3.582927426513e-15     3.799296636507e-15     1.301393180595e-04
          5.58108540       5.96666863e-16    -7.590425251989e-15    -8.370429586246e-15     1.351360319206e-04
          5.78779227       1.97362907e-16     2.731661276176e-15     9.385722934298e-16     1.401118497031e-04
          5.99449913      -4.11529977e-16     6.618597444388e-16     7.668771576101e-16     1.450599261442e-04
          6.20120600       5.28497535e-16    -3.513040838462e-15    -2.702656466824e-15     1.499718799309e-04
          6.40791287      -5.92115184e-16     6.452111505651e-15     5.226408269066e-15     1.548379555140e-04
          6.61461974       7.14772341e-17     1.401899118683e-15     2.686153730468e-15     1.596472317913e-04
          6.82132660       6.25353024e-16    -4.133101985515e-15    -6.440484577764e-15     1.643878375064e-04
          7.02803347       7.87017009e-17    -5.983700504045e-15    -2.674485129005e-15     1.690471418720e-04
          7.23474034       8.25679815e-16    -8.821379359232e-15    -7.716196106335e-15     1.736119057346e-04
          7.44144720      -1.23609834e-16     3.600472002284e-15     4.318132665865e-15     1.780683938350e-04
          7.64815407       2.31421108e-16    -3.009874073926e-15    -2.952156170259e-15     1.824024591310e-04
          7.85486094       5.90659706e-17    -2.678194077327e-15    -1.631395505323e-15     1.865996144083e-04
          8.06156780       9.83665408e-16    -6.524290140418e-15    -7.669813415517e-15     1.906451044710e-04
          8.26827467       7.22023271e-16    -7.297126619489e-15    -7.529956892262e-15     1.945239896133e-04
          8.47498154       1.14448257e-15    -1.517109888704e-14    -1.305641491325e-14     1.982212454194e-04
          8.68168840       5.41887939e-16    -7.557253084971e-15    -6.601219562923e-15     2.017218816775e-04
          8.88839527       7.70583332e-16    -1.013968128314e-14    -9.344966176329e-15     2.050110790514e-04
          9.09510214       4.58555169e-16    -7.639683419596e-15    -6.865805101107e-15     2.080743412339e-04
          9.30180900       2.78049352e-16    -2.204865593676e-15    -2.871642820161e-15     2.108976581326e-04
          9.50851587       9.55005710e-16    -4.247287585551e-15    -1.382854294151e-15     2.134676752817e-04
          9.71522274      -5.30985080e-16     4.293253540602e-15     6.855303359790e-17     2.157718638836e-04
          9.92192960       9.34152671e-18    -2.481244754054e-16    -1.176028333211e-16     2.177986853620e-04
         10.12863647       1.86883461e-16     3.294963011825e-15     2.368142667012e-15     2.195377445833e-04
         10.33534334       3.38147389e-16    -6.359096082556e-15    -4.793336460465e-15     2.209799253271e-04
         10.54205020       1.09621360e-15    -1.091985231171e-14    -8.705360121824e-15     2.221175024362e-04
         10.74875707       4.81842828e-16    -7.508453326708e-15    -4.110848295581e-15     2.229442253439e-04
         10.95546394      -4.21559548e-16     5.946819388706e-17     1.830803569619e-15     2.234553688999e-04
         11.16217080      -7.67487132e-16     5.294169504708e-15     6.194943864111e-15     2.236477485392e-04
         11.36887767       2.90725248e-16    -4.044712329430e-15    -3.004206467501e-15     2.235196985220e-04
         11.57558454       4.48525598e-16    -6.050461573847e-15    -2.963616403839e-15     2.230710137043e-04
         11.78229140       4.45746957e-16    -3.979118119774e-15     9.657017918323e-16     2.223028568634e-04
         11.98899827       2.92154263e-16    -3.110349067242e-15    -5.644935999415e-15     2.212176355467e-04
         12.19570514      -2.81965912e-16    -1.103432962678e-15     4.038169577895e-17     2.198188535128e-04
         12.40241200       2.63732734e-16    -6.147519333877e-15    -5.081800958072e-15     2.181109426140e-04
         12.60911887      -2.92948160e-16    -2.249039584931e-15     1.788629910044e-16     2.160990814844e-04
         12.81582574       6.58485011e-16    -6.327382490721e-15    -9.660935234529e-15     2.137890070594e-04
         13.02253260       2.87576121e-16    -5.290418882809e-15    -1.529128548211e-15     2.111868250281e-04
         13.22923947       2.46849182e-16    -4.351888262955e-15    -1.266209953093e-15     2.082988240720e-04
         13.43594634       6.66609229e-17    -5.127766913123e-15    -3.231785869615e-15     2.051312986552e-04
         13.64265320      -1.12680512e-16     4.056297583740e-15     8.769079020530e-15     2.016903841875e-04
         13.84936007       2.76805579e-16    -6.306170640204e-15    -5.198028563364e-15     1.979819076193e-04
         14.05606694       5.53478842e-16    -3.961656891156e-15    -5.941651865200e-15     1.940112562669e-04
         14.26277380       5.19420641e-16    -7.297585028833e-15    -1.008775600663e-14     1.897832668642e-04
         14.46948067       1.22763010e-16    -4.031168417017e-15    -4.190319806263e-15     1.853021366287e-04
         14.67618754      -2.23376280e-16    -2.689612637330e-16     2.673026553822e-15     1.805713576031e-04
         14.88289440      -6.60390365e-16     6.488617558801e-15     9.401058810507e-15     1.755936751194e-04
         15.08960127       7.93315262e-16    -6.862679582858e-15    -8.559794318567e-15     1.703710709089e-04
         15.29630814       2.21920802e-16    -2.029211468075e-15    -4.763956588923e-15     1.649047710759e-04
         15.50301500      -5.31514345e-16     6.001828509891e-15     7.227656767205e-15     1.591952784239e-04
         15.70972187      -7.55525743e-17    -2.053048753921e-15    -1.221035795999e-16     1.532424289670e-04
         15.91642874       5.60570992e-16    -8.215278860358e-15    -6.134475504384e-15     1.470454715874e-04
         16.12313560       1.09324972e-15    -1.002028648603e-14    -8.628639067202e-15     1.406031696383e-04
         16.32984247       6.69890671e-16    -5.730075116522e-15    -3.839469964402e-15     1.339139230198e-04
         16.53654934      -3.69162316e-16    -2.715033519090e-15    -1.136021699622e-16     1.269759084297e-04
         16.74325620       4.33229841e-16    -6.723281468946e-15    -3.971700223130e-15     1.197872349760e-04
         16.94996307       8.23800925e-16    -6.170814863228e-15    -4.187486003050e-15     1.123461113661e-04
         17.15666994       2.79769463e-16    -4.511081325779e-15    -1.839013264220e-15     1.046510200920e-04
         17.36337680       6.99635363e-16    -2.572176498316e-15    -4.456655634667e-15     9.670089394579e-05
         17.57008367      -9.54794004e-17    -5.306171494784e-15    -4.821007715364e-15     8.849528976036e-05
         17.77679054       8.38884976e-17     7.618346548348e-16     1.929069863372e-16     8.003455441378e-05
         17.98349740       3.52252300e-16    -5.491743931630e-16    -3.252956046556e-15     7.131997874053e-05
         18.19020427       3.13854127e-17     4.212365128314e-15     4.300921478706e-15     6.235393500236e-05
         18.39691114       1.21889722e-16     4.671232880865e-15     7.824589079215e-15     5.313999437275e-05
         18.60361800       5.92512133e-16    -1.198440382737e-14    -8.636765414649e-15     4.368302142949e-05
         18.81032487      -1.90932338e-16     3.386478186160e-15     1.261125776741e-15     3.398924316626e-05
         19.01703174      -3.69956213e-17     1.258750382872e-15     2.148939653805e-15     2.406629051337e-05
         19.22373861       6.14238460e-16    -9.242782566370e-16    -1.481037240751e-15     1.392321176038e-05
         19.43044547      -1.23239348e-16    -7.605844475351e-16    -4.075259061118e-16     3.570457104202e-06
         19.63715234       8.08346387e-16    -5.840718462541e-15    -4.767040433595e-15    -6.980165101233e-06
         19.84385921      -2.01173615e-16     1.405066310509e-15     1.106183418737e-15    -1.771555913585e-05
         20.05056607       4.53818248e-16     1.577761612168e-15     1.117685325894e-15    -2.862143823077e-05
         20.25727294       5.49747523e-16    -6.548710856338e-15    -6.111388342917e-15    -3.968244356746e-05
         20.46397981      -3.58206531e-16     8.271079779499e-15     9.841465168600e-15    -5.088228340078e-05
         20.67068667       6.62322182e-16    -6.232325062371e-15    -3.222034252678e-15    -6.220388858695e-05
'''.strip())  # noqa: E501

    check_txt_data('dm.dat', 'dm_ref.dat', atol=1e-8)

    """
    Run in 2 splits, check if restart still works.
    IMPORTANT: The simple convolution implemented at the moment is
    heavility dependent on the time-steppin dt. With the large dt used here,
    the deviation between full run and restarted is quite large. If you reduce
    dt to a more reasonable value and add more steps, the error goes quickly
    down. I tried with 0.1 spacing and same total time and the error at the end
    was on the order of 1e-7, which seems alright for this method.
    """

    td_calc = LCAOTDDFT('gs.gpw',
                        rremission=RRemission(cross_section, [0, 0, 1],
                                              [2.98, 3, 0.01, 10, 5000, 60000],
                                              [1, 1, 0, 0, 0]))
    DipoleMomentWriter(td_calc, 'dm_split.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(dt, int(nt * 0.5))
    td_calc.write('td_split0.gpw', mode='all')
    td_calc_restart = LCAOTDDFT('td_split0.gpw')
    DipoleMomentWriter(td_calc_restart, 'dm_split.dat')
    td_calc_restart.propagate(dt, int(nt * 0.5))
    td_calc_restart.write('td_split1.gpw', mode='all')

    with paropen('dm_ref_restart.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000       1.44357020e-16    -1.055049940181e-15    -1.183029494090e-15    -2.047822887409e-14
# Kick = [    0.000000000000e+00,     0.000000000000e+00,     1.000000000000e-05]; Time = 0.00000000
          0.00000000      -4.80519666e-16     2.313133545827e-15    -3.058840526496e-16    -2.173477055740e-14
          0.20670687       4.13435331e-16    -4.824758337263e-15    -2.693988362879e-15     4.732577230067e-06
          0.41341373       1.48273581e-16    -1.802590558223e-15    -7.918813036032e-16     9.446647009121e-06
          0.62012060       4.73559831e-16    -4.846970353620e-15    -1.021752752434e-15     1.416965327739e-05
          0.82682747      -5.22860863e-16     7.783915668399e-15     4.545753741556e-15     1.894481682408e-05
          1.03353433      -4.16002266e-17    -5.875599246202e-15    -4.228701170363e-15     2.379312470542e-05
          1.24024120       5.57792351e-16    -6.720614360040e-15    -9.846299303492e-15     2.871198789831e-05
          1.44694807       6.88256166e-16    -5.888393034235e-15    -7.676897923549e-15     3.369018041419e-05
          1.65365493       9.05069561e-16    -3.930068320051e-15    -1.834637538671e-15     3.871688287533e-05
          1.86036180       1.81405568e-16    -5.065631610330e-15    -6.921731040979e-15     4.378217907209e-05
          2.06706867       8.60055575e-18    -2.926193532002e-15    -8.168021024431e-16     4.887691006699e-05
          2.27377553       6.33318462e-16    -2.826552010220e-15    -4.055880847973e-15     5.399275444136e-05
          2.48048240      -6.98153421e-16     4.676775466560e-15     4.813131409376e-15     5.912232932113e-05
          2.68718927       1.49332111e-16     2.219201304046e-15     2.755123499833e-15     6.425932654607e-05
          2.89389613       5.95026142e-16    -8.792457897033e-15    -6.209404595210e-15     6.939858405594e-05
          3.10060300       4.14467397e-16    -3.922692096983e-15    -2.549631093345e-15     7.453610113963e-05
          3.30730987       5.72426527e-16    -9.478529989509e-15    -7.853218826377e-15     7.966901744572e-05
          3.51401673       1.67882848e-16    -3.100305735268e-15    -8.071755062357e-15     8.479554502827e-05
          3.72072360      -1.51634414e-17     4.299046167757e-16     3.302005846280e-15     8.991485283752e-05
          3.92743047       3.32643033e-16    -6.160646510524e-15    -6.643143181038e-15     9.502690551057e-05
          4.13413733       5.95423090e-16    -5.965155762434e-15    -1.033183814510e-14     1.001322595198e-04
          4.34084420       8.49470276e-16    -8.849300655591e-15    -7.081799248912e-15     1.052318180846e-04
          4.54755107       5.36039561e-16    -1.394022812695e-15     4.054839008557e-16     1.103265483827e-04
          4.75425793       4.65276834e-16    -5.318256832014e-15    -4.077217719221e-15     1.154171677792e-04
          4.96096480      -5.41173431e-17    -3.171609224925e-15    -4.204155433713e-15     1.205038129893e-04
          5.16767167      -5.92485670e-16     5.073841304931e-15     4.536418860385e-15     1.255857122159e-04
          5.37437853       3.46483312e-16    -6.486992289311e-15    -2.195197323892e-15     1.306608883241e-04
          5.58108540       4.23411975e-16    -8.919395611525e-16    -3.273376099117e-15     1.357259245907e-04
          5.78779227       1.53936716e-16    -2.072718682103e-15     3.186778406828e-16     1.407758227126e-04
          5.99449913       5.19473567e-17    -2.879727494032e-15    -3.570258659209e-15     1.458039709587e-04
          6.20120600       2.17183880e-16    -2.671401284332e-15    -1.938154703083e-15     1.508022206118e-04
          6.40791287       1.27341152e-16    -2.106932688537e-15    -3.280418933572e-15     1.557610473091e-04
          6.61461974       5.54140423e-16    -2.203698733530e-15    -1.366059842759e-15     1.606697591436e-04
          6.82132660       4.58184684e-16    -6.028582946103e-15    -7.711695360056e-15     1.655167112253e-04
          7.02803347       1.52692944e-17     1.139480606485e-15     1.981911958571e-15     1.702894954735e-04
          7.23474034       7.31338334e-16    -7.021205868455e-15    -5.868348043864e-15     1.749750908039e-04
          7.44144720       1.10169150e-15    -1.680482811266e-14    -1.598535890131e-14     1.795599743424e-04
          7.64815407       2.70771958e-16    -3.103931336437e-15    -8.665603529695e-16     1.840302046623e-04
          7.85486094      -7.42294119e-16     5.433442597889e-15     9.366178026846e-15     1.883714920874e-04
          8.06156780       8.58997045e-17    -3.662607305077e-15    -3.589553525201e-15     1.925692698120e-04
          8.26827467       4.85415366e-16    -7.772622129126e-15    -6.866346857604e-15     1.966087760141e-04
          8.47498154       7.60315591e-16    -5.617598133129e-15    -8.105844048061e-15     2.004751524087e-04
          8.68168840       3.81044315e-16    -2.999372332609e-15    -5.112431036914e-15     2.041535617369e-04
          8.88839527       1.16573256e-15    -1.492551649981e-14    -1.118522964785e-14     2.076293231817e-04
          9.09510214      -5.22225745e-16     3.233161097645e-15     5.631642128462e-15     2.108880631484e-04
          9.30180900       2.55661443e-16    -3.933860615527e-15    -6.385933865922e-15     2.139158769782e-04
          9.50851587      -6.77459161e-18    -3.166775090033e-15    -1.436779902343e-15     2.166994971983e-04
          9.71522274      -1.03222547e-15     2.236454164781e-15     2.013250488216e-15     2.192264620778e-04
          9.92192960       8.14829883e-16    -6.338425988535e-15    -7.204236217125e-15     2.214852789987e-04
         10.12863647       5.14048601e-16    -1.052757893465e-15    -2.178944628997e-15     2.234655764899e-04
         10.33534334       1.33779660e-15    -1.260013092245e-14    -1.207721088258e-14     2.251582384839e-04
# Start; Time = 10.33534334
         10.33534334       1.05980017e-15    -1.019118982389e-14    -9.405726251092e-15     2.251582384841e-04
         10.54205020       7.09559081e-16    -1.151720135948e-14    -1.078633017210e-14     2.265479242503e-04
         10.74875707       6.41310363e-16    -3.780626874165e-15    -2.662108076738e-15     2.276125959687e-04
         10.95546394      -1.00772050e-16    -3.176360012664e-16    -2.052840386038e-16     2.283418504395e-04
         11.16217080       5.34028354e-16    -7.336799864465e-15    -5.438110038475e-15     2.287403121111e-04
         11.36887767       4.42650757e-16    -4.891394386334e-15    -6.444276873239e-15     2.288145878681e-04
         11.57558454       2.31156475e-16    -5.416731493651e-16    -1.359308723341e-15     2.285674477746e-04
         11.78229140      -1.72540380e-16    -2.404565373008e-17     2.350598091240e-15     2.279998574997e-04
         11.98899827       8.31237097e-16    -8.520787850818e-15    -8.266495686067e-15     2.271139026386e-04
         12.19570514      -6.69520186e-18    -3.685694466544e-15    -4.591052919380e-15     2.259128595407e-04
         12.40241200       3.69426948e-17    -5.679275026579e-16    -4.441986535684e-16     2.244009356318e-04
         12.60911887       7.17471592e-16    -4.442486618604e-15    -2.546297207213e-15     2.225830692829e-04
         12.81582574       6.78147205e-16    -6.203403600172e-15    -5.459280215416e-15     2.204647166821e-04
         13.02253260       3.17294349e-16    -5.699528384833e-15    -3.994787384597e-15     2.180516728438e-04
         13.22923947       1.04926780e-16     2.987245321802e-15     2.659190926372e-15     2.153498931781e-04
         13.43594634       1.02817659e-15    -6.802377917438e-15    -7.372305751775e-15     2.123653170065e-04
         13.64265320       1.44841298e-15    -1.359562932159e-14    -1.201636746066e-14     2.091037077691e-04
         13.84936007       1.74630977e-16    -2.202156811194e-15    -4.226659165106e-15     2.055705099913e-04
         14.05606694       7.52535396e-16    -1.096315115785e-14    -8.220071321673e-15     2.017707250726e-04
         14.26277380       1.00705892e-15    -9.256243131630e-15    -4.766206962062e-15     1.977088080591e-04
         14.46948067       1.10068589e-15    -1.101007560517e-14    -8.792457897033e-15     1.933885876384e-04
         14.67618754       2.55423274e-16    -2.285879026694e-15    -1.105850030124e-15     1.888132104256e-04
         14.88289440       3.14515708e-16    -2.392021626435e-15    -1.860141767584e-15     1.839851092377e-04
         15.08960127       3.31134628e-16    -9.695816018189e-15    -9.187690098032e-15     1.789059936430e-04
         15.29630814       9.75964603e-17    -3.328260199572e-15     1.100807527349e-15     1.735768604755e-04
         15.50301500       6.36335272e-16    -9.352717461587e-15    -4.091511756013e-15     1.679980213984e-04
         15.70972187       5.25480725e-16    -5.838134700789e-15    -2.385562222054e-15     1.621691454864e-04
         15.91642874      -5.63905361e-16     2.878477286732e-15    -5.618431604662e-16     1.560893162667e-04
         16.12313560       6.09977877e-16    -6.779624144584e-15    -4.984493156582e-15     1.497571059516e-04
         16.32984247      -3.47356599e-16    -2.557965808677e-15    -9.104843027641e-16     1.431706719349e-04
         16.53654934      -1.00957293e-16     1.310842353691e-15     8.020913298838e-16     1.363278801644e-04
         16.74325620       7.44146547e-16    -8.771912823742e-15    -7.807086177020e-15     1.292264576973e-04
         16.94996307       6.09819097e-16    -3.001330990712e-15    -3.589303483741e-15     1.218641709402e-04
         17.15666994      -1.22313134e-16     1.766292872957e-15    -9.310293760552e-16     1.142390214033e-04
         17.36337680       4.21347842e-16    -3.326801624389e-15    -1.507333267621e-15     1.063494485128e-04
         17.57008367       1.05336960e-15    -8.773163031042e-15    -8.637098803263e-15     9.819452846852e-05
         17.77679054       4.28175360e-17    -4.285335561037e-15    -7.413895981277e-15     8.977416055866e-05
         17.98349740       6.65762405e-16    -5.898269671902e-15    -5.654854310659e-15     8.108923477885e-05
         18.19020427       4.06052084e-16    -4.190403153416e-15    -4.258997860591e-15     7.214177707814e-05
         18.39691114       8.93055246e-16    -8.363095036754e-15    -3.029335634224e-15     6.293506950345e-05
         18.60361800       3.93005703e-16    -9.227363343008e-15    -4.170024774432e-15     5.347374384084e-05
         18.81032487       1.01168999e-15    -1.080237449912e-14    -5.930358325927e-15     4.376384718083e-05
         19.01703174      -8.96389615e-16     6.814921664011e-15     6.716405328798e-15     3.381287883832e-05
         19.22373861       6.59199519e-16    -5.529916927846e-15    -3.430568830261e-15     2.362979777185e-05
         19.43044547       3.68606588e-16    -4.627975708297e-15    -7.149185422363e-15     1.322500121711e-05
         19.63715234       2.60662997e-16    -7.216613269391e-16     2.069093080934e-16     2.610274568655e-06
         19.84385921       3.37327028e-16    -8.212069994956e-15    -5.403062560508e-15    -8.201285495375e-06
         20.05056607      -1.34062817e-16    -3.950655066919e-16    -4.605346956173e-16    -1.919537280468e-05
         20.25727294       2.48225271e-16    -3.493204215974e-15    -2.297630975311e-15    -3.035659590170e-05
         20.46397981       3.06047468e-16    -7.446234676762e-16    -4.202738532106e-15    -4.166862109650e-05
         20.67068667       1.06376966e-15    -1.279512158762e-14    -1.582633253279e-14    -5.311433121742e-05
'''.strip())  # noqa: E501

    check_txt_data('dm_split.dat', 'dm_ref_restart.dat', atol=1e-8)
