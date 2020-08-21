from gpaw.atom.generator2 import generate


def test_lithium(in_tmp_dir):
    G = generate('Li', 3, '2s,2p,s', [2.1, 2.1], 2.0, None, 2, 'PBE', True)
    assert G.check_all()
    basis = G.create_basis_set()
    basis.write_xml()
    setup = G.make_paw_setup('test')
    setup.write_xml()


def test_pseudo_h(in_tmp_dir):
    G = generate('H', 1.25, '1s,s', [0.9], 0.7, None, 2, 'PBE', True)
    assert G.check_all()
    basis = G.create_basis_set()
    basis.write_xml()
    setup = G.make_paw_setup('test')
    setup.write_xml()
