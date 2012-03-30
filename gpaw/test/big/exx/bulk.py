from ase.dft.kpoints import monkhorst_pack

from gpaw.test.big.exx.hsk import HarlSchimkaKresseEXXBulkTask as EXXBulkTask

from gpaw import PW
from gpaw.factory import GPAWFactory


for ecut in [340]:
    for k in [6]:
        kpts = monkhorst_pack((k, k, k)) + 0.5 / k
        task = EXXBulkTask(
            calcfactory=GPAWFactory(xc='PBE', 
                                    mode=PW(ecut),
                                    kpts=kpts),
            tag='%d-%d' % (ecut, k),
            use_lock_files=True)
        task.run()
