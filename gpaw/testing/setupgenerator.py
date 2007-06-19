#!/usr/bin/env python

import os
from gpaw.atom import generator
#from generator import Generator
Generator = generator.Generator

class SetupGenerator:
    """
    A SetupGenerator generates setups for a particular element during a
    setup optimization run with an Optimizer. The generated setup files
    will be named according to the name of this SetupGenerator.

    Any existing setup with the given element/name combination will be
    overwritten. Each concurrently active generator should thus use a
    different name or element.

    A SetupGenerator must have a standard_parameters() method which is
    used to initialize the optimizer.

    A SetupGenerator should possess a method which generates a setup given
    some number of parameters in a way compatible with the
    standard_parameters() method.
    """
    def __init__(self, symbol, name):
        """
        Creates a SetupOptimizer for the element with the given symbol
        (string), where setup files will be generated with the name
        <symbol>.<name>.PBE . The name parameter should be a non-empty
        string.
        """
        #We don't want anything to mess up with existing files
        #so make sure a proper name is entered with a couple of chars
        #(it should be enough to test for len==0, but what the heck)
        if len(name) < 1:
            raise Exception('Please supply a non-empty name.')
        self.symbol = symbol
        self.name = name


    #def new_nitrogen_setup(self, r=1.1, rvbar=None, rcomp=None, 
    #                       rfilter=None, hfilter=0.4):
        """Generate new nitrogen setup.

        The new setup depends on five parameters (Bohr units):

        * 0.6 < r < 1.9: cutoff radius for projector functions
        * 0.6 < rvbar < 1.9: cutoff radius zero potential (vbar)
        * 0.6 < rcomp < 1.9: cutoff radius for compensation charges
        * 0.6 < rfilter < 1.9: cutoff radius for Fourier-filtered
          projector functions
        * 0.2 < hfilter < 0.6: target grid spacing

        Use the setup like this::

          calc = Calculator(setups={'N': name}, ...)

        where name is the name of this SetupGenerator
        """

        #if rvbar is None:
        #    rvbar = r
        #if rcomp is None:
        #    rcomp = r
        #if rfilter is None:
        #    rfilter = 2 * r

        #g = Generator(self.symbol, 'PBE', scalarrel=True, nofiles=True)
        #g.run(core='[He]',
        #      rcut=r,
        #      vbar=('poly', rvbar),
        #      filter=(hfilter, rfilter / r),
        #      rcutcomp=rcomp,
        #      logderiv=False)
        #path = os.environ['GPAW_SETUP_PATH'].split(':')[0]
        #os.rename(symbol+'.PBE', path + '/'+symbol+'.'+self.name+'.PBE')
        #return # Without return here the editor can't figure out the correct indentation

    def standard_parameters(self, r=None, rvbar=None, rcomp=None,
                           rfilter=None, hfilter=0.4):
        """
        Given up to five argument variables, selects any remaining parameters
        and returns them in a quintuple.

        The rcut-parameter is selected per default from the dictionary
        gpaw.atom.generator.parameters. Any remaining unspecified parameters
        are set set in terms of rcut except hfilter which deafults to 0.4.
        """

        param = generator.parameters[self.symbol]
        if r is None:
            r = param['rcut']
        if rvbar is None:
            rvbar = r
        if rcomp is None:
            rcomp = r
        if rfilter is None:
            rfilter = 2 * r

        return (r, rvbar, rcomp, rfilter, hfilter)

        
    def new_setup(self, r=None, rvbar=None, rcomp=None, 
                           rfilter=None, hfilter=0.4):

        """Generate new molecule setup.

        The new setup depends on five parameters (Bohr units):

        * 0.6 < r < 1.9: cutoff radius for projector functions
        * 0.6 < rvbar < 1.9: cutoff radius zero potential (vbar)
        * 0.6 < rcomp < 1.9: cutoff radius for compensation charges
        * 0.6 < rfilter < 1.9: cutoff radius for Fourier-filtered
          projector functions
        * 0.2 < hfilter < 0.6: target grid spacing

        Use the setup like this::

          calc = Calculator(setups={'N': 'opt'}, ...)

        """

        (r, rvbar, rcomp, rfilter,hfilter) = \
            self.standard_parameters(r, rvbar, rcomp, rfilter, hfilter)
                                                                       

        param = generator.parameters[self.symbol]

        core=''
        if param.has_key('core'):
            core=param['core']
        
        g = Generator(self.symbol, 'PBE', scalarrel=True, nofiles=True)
        g.run(core=core,
              rcut=r,
              vbar=('poly', rvbar),
              filter=(hfilter, rfilter / r),
              rcutcomp=rcomp,
              logderiv=False)
        path = os.environ['GPAW_SETUP_PATH'].split(':')[0]
        os.rename(self.symbol+'.PBE', path + '/'+self.symbol+'.'+self.name+
                  '.PBE')

    def generate_setup(self, par):
      """
      Calls new_setup with after unpacking a parameter list. This is a
      convenience method which is invoked directly by the Optimizer.
      """
      self.new_setup(*par)
