from gpaw.core.plane_waves import PWArray, PWDesc
from gpaw.new.pwfd import ArrayCollection, LBFGS


def test_initialize_and_update():
    wfs_u: list[PWArray] = []
    for kpt in [(1, 1, 1), (2, 2, 2)]:
        pwdesc = PWDesc(ecut=5, cell=[1, 1, 1], kpt=kpt, dtype=complex)
        wfs_u.append(pwdesc.empty())
    optimizer = LBFGS()
    optimizer.update(ArrayCollection(wfs_u))
