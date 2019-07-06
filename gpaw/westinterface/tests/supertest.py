from testframework import BaseTester
from gpaw.westinterface import WESTInterface
from ase import Atoms
from gpaw import GPAW, PW

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=(4,4,4))

calc = GPAW(mode=PW(100), txt="supertestout.txt")
atoms.set_calculator(calc)
atoms.get_potential_energy()


wi = WESTInterface(calc, atoms=atoms, use_dummywest=True)

class Tester(BaseTester):

    def test_01_instancefrompurecalc(self):
        lwi = WESTInterface(calc)
        assert lwi.atoms is not None
        assert lwi.calc is not None

    def test_02_noatomsfails(self):
        try:
            lcalc = GPAW(mode=PW(100), txt=None)
            lwi = WESTInterface(lcalc)
            raise ValueError("Instantiation should have failed")
        except:
            pass

    def test_03_nocalcfails(self):
        try:
            lwi = WESTInterface(None)
            raise Exception("Instantiation should have failed")
        except ValueError:
            pass

    def test_04_dryrunsgbar(self):
        # Check that submit strings are correct for various params
        # Prints to console and/or returns

        lwi = WESTInterface(calc, atoms=atoms, computer="gbar", use_dummywest=True)

        opt_opts = [{'Time': '05:00:00',
                     'GPAWNodes': 5,
                     'WESTNodes': 5,
                     'JobName': 'TestName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     },
                    {'Time': '05:00:00',
                     'GPAWNodes': 5,
                     'WESTNodes': 20,
                     'JobName': 'OTestname',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     },
                    {'Time': '00:10:15',
                     'GPAWNodes': 5,
                     'WESTNodes': 5,
                     'JobName': 'TestName',
                     'Input': 'inputname.xml',
                     'Output': 'outputgame.xml',
                     'Calcname': 'calcname.gpw'
                     },
                    {'Time': '05:00:00',
                     'GPAWNodes': 10,
                     'WESTNodes': 10,
                     'JobName': 'MyName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'mycalcname.gpw'
                     }
                    ]
        
        for i, opts in enumerate(opt_opts):
            cmd = lwi.run(opts, dry_run=True)
            with open("gbarsub{}.txt".format(i), "r") as f:
                expected_cmd = f.read()
            assert cmd == expected_cmd
            
    def test_05_dryrunsniflheim(self):
        lwi = WESTInterface(calc, atoms=atoms, computer="niflheim", use_dummywest=True)

        opt_opts = [{'Time': '05:00:00',
                     'GPAWNodes': 5,
                     'WESTNodes': 5,
                     'Partition': 'xeon8',
                     'JobName': 'TestName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     },
                    {'Time': '05:00:00',
                     'GPAWNodes': 5,
                     'WESTNodes': 20,
                     'Partition': 'xeon8',
                     'JobName': 'OTestName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     },
                    {'Time': '00:10:15',
                     'GPAWNodes': 5,
                     'WESTNodes': 5,
                     'Partition': 'xeon24',
                     'JobName': 'TestName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     },
                    {'Time': '05:00:00',
                     'GPAWNodes': 5,
                     'WESTNodes': 10,
                     'Partition': 'xeon24',
                     'JobName': 'TestNameName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     }
                    ]
        
        for i, opts in enumerate(opt_opts):
            cmd = lwi.run(opts, dry_run=True)
            with open("niflheimsub{}.txt".format(i), "r") as f:
                expected_cmd = f.read()
            b = cmd == expected_cmd

            if not b:
                for i in range(len(cmd)):
                    if cmd[i] != expected_cmd[i]:
                        print("EXP: ", expected_cmd[:i+1])
                        print("ACT: ", cmd[:i+1])
                        break
            assert b, expected_cmd[-5:] + "---" + cmd[-5:]

    def test_06_instancefromnokw(self):
        lwi = WESTInterface(calc, atoms)

    def test_07_writesinputfiles(self):
        opts = {'Time': '05:00:00',
                     'GPAWNodes': 5,
                     'WESTNodes': 5,
                     'Partition': 'xeon8',
                     'JobName': 'TestName',
                     'Input': 'inputname.xml',
                     'Output': 'outputname.xml',
                     'Calcname': 'calcname.gpw'
                     }
        wi.run(opts, dry_run=True)
        import os
        assert os.path.exists(opts["Input"])
 

if __name__ == "__main__":
    tester = Tester()
    tester.run_tests()
