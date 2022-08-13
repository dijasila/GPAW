import asr
from ase.build import bulk


@asr.workflow
class MyWorkflow:
    atoms = asr.var()
    calculator = asr.var()

    @asr.task
    def relax(self):
        return asr.node('relax', atoms=self.atoms, calculator=self.calculator)

    # --- end-snippet-1 ---

    @asr.task
    def groundstate(self):
        return asr.node('groundstate', atoms=self.relax,
                        calculator=self.calculator)

    @asr.task
    def bandstructure(self):
        return asr.node('bandstructure', gpw=self.groundstate)


#def workflow(rn):
#    calculator = {'mode': 'pw',
#                  'kpts': (4, 4, 4),
#                  'txt': 'gpaw.txt'}
#    wf = MyWorkflow(atoms=bulk('Si'), calculator=calculator)
#    rn.run_workflow(wf)


def workflow():
    return MyWorkflow(
        atoms=bulk('Si'),
        calculator={'mode': 'pw',
                    'kpts': (4, 4, 4),
                    'txt': 'gpaw.txt'})
