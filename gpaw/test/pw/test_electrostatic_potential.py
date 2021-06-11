from gpaw import GPAW


def test_electrostatic_potential(gpw_files):
    calc = GPAW(gpw_files['h2_pw'], parallel=dict(domain=2))
    v = calc.get_electrostatic_potential()
    print(v.shape)
