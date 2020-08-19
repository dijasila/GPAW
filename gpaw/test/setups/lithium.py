from gpaw.atom.generator2 import generate


def test_lithium():
    G = generate('Li', 3, '2s,2p,s', [2.1, 2.1], 2.0, 0.0, 2, 'PBE', True)
    assert G.check_all()
    basis = G.create_basis_set()
    basis.write_xml()
    setup = G.make_paw_setup('test')
    setup.write_xml()


def test_pseudo_h():
    G = generate('H', 1.25, '1s,s', [0.9], 0.7, 0.0, 2, 'PBE', True)
    assert G.check_all()
    basis = G.create_basis_set()
    basis.write_xml()
    setup = G.make_paw_setup('test')
    setup.write_xml()
