import gpaw.mpi as mpi
from gpaw.response.pair import get_PairMatrixElement


class ChiKS:
    """Class for calculating response functions in the Kohn-Sham system"""

    def __init__(self, gs, response='susceptibility',
                 world=mpi.world, txt='-', timer=None, **kwargs):
        """Construct the ChiKS object

        Currently only collinear Kohn-Sham systems are supported

        Parameters
        ----------
        gs : str
            The groundstate calculation file that the linear response
            calculation is based on.
        response : str
            Type of response function.
            Currently, only susceptibilities are supported.
        world : obj
            MPI communicator.
        txt : str
            Output file.
        timer : func
            gpaw.utilities.timing.timer wrapper instance
        """

        self.response = response

        # Initialize the PairMatrixElement object
        PME = get_PairMatrixElement(response)
        self.pair = PME(gs, world=world, txt=txt, timer=timer, **kwargs)

        # Extract ground state calculator, timer and filehandle for output
        calc = self.pair.calc
        self.calc = calc
        self.timer = self.pair.timer
        self.fd = self.pair.fd
