import numpy as np

import gpaw.mpi as mpi


class Integrator():

    def __init__(self, calc, world=None, nblocks=1,
                 blockcomm=None, kncomm=None,
                 txt=None, timer=None):
        """Baseclass, performing Brillouin Zone integration, summing a given integrand
        over bands and spin.
        
        calc : obj
            GPAW calculator object
        world : obj
            MPI communicator.
        blockcomm : obj
            MPI communicator
        kncomm : obj
            MPI communicator
        nblocks : int
            Divide the response function storage into nblocks. Useful when the
            response function is large and memory requirements are restrictive.
        txt : str
            Output file.
        timer : func
            gpaw.utilities.timing.timer wrapper instance
        """
        

        self.calc = calc

        self.world = world
        self.nblocks = nblocks

        if world.rank != 0:
            txt = devnull
        self.fd = convert_string_to_fd(txt, world)

        self.timer = timer or Timer()

    def distribute_domain(self, domain_dl):
        """Distribute integration domain. """
        domainsize = [len(domain_l) for domain_l in domain_dl]
        nterms = np.prod(domainsize)
        size = self.kncomm.size
        rank = self.kncomm.rank

        n = (nterms + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, nterms)
        mydomain = []
        for i in range(i1, i2):
            unravelled_d = np.unravel_index(i, domainsize)
            arguments = []
            for domain_l, index in zip(domain_dl, unravelled_d):
                arguments.append(domain_l[index])
            mydomain.append(tuple(arguments))

        print('Distributing domain %s' % (domainsize, ),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

        return mydomain

    def integrate(self, *args, **kwargs):
        raise NotImplementedError
