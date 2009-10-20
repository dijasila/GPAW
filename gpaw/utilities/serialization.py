import sys
from optparse import OptionParser, OptionGroup
from gpaw.parameters import InputParameters
from gpaw.poisson import PoissonSolver


def build_parser():
    usage = '%prog [OPTIONS] [FILE]'
    description = ('Print representation of GPAW input parameters to stdout.')
    p = OptionParser(usage=usage, description=description)
    return p

def append_to_optiongroup(parameters, opts):
    for key, value in parameters.items():
        opts.add_option('--%s' % key, default=repr(value), type=str,
                        help='default=%default')
    return opts


def deserialize(filename):
    """Get an InputParameters object from a filename."""
    stringrep = open(filename).read()
    if not stringrep.startswith('InputParameters(**{'):
        raise ValueError('Does not appear to be a serialized InputParameters')
    parameters = eval(stringrep)
    return parameters

def stringdict2inputparameters(parameters):
    """Convert values from string representations to objects.

    Returns an InputParameters given an ordinary dictionary."""
    reconstructed_parameters = {}
    for key, value in parameters.items():
        reconstructed_parameters[key] = eval(value)
    output = InputParameters(**reconstructed_parameters)
    return output


def populate_parser(parser, defaultparameters):
    opts = OptionGroup(parser, 'GPAW parameters')
    append_to_optiongroup(defaultparameters, opts)
    parser.add_option_group(opts)
    return parser
    opts, args = parser.parse_args()
    return opts, args


def main():
    # build from scratch
    # build from existing
    # build from gpw?
    # just print nicely?
    # print as python script?
    parser = build_parser()
    populate_parser(parser, InputParameters())
    opts, args = parser.parse_args()
    
    if len(args) == 1:
        deserialized_parameters = deserialize(args[0])
        # We have to use the newly loaded info (target file)
        # to get new defaults!
        #
        # Rather ugly but hopefully it works
        # We can probably avoid this somehow, think about it in the future
        # (can one have different call formats like e.g. the 'du' man-page,
        # and somehow detect which one is relevant?)
        parser2 = OptionParser()
        populate_parser(parser2, deserialized_parameters)
        opts, args2 = parser2.parse_args()
        
    parameters = {}
    parameters.update(vars(opts))
    output = stringdict2inputparameters(parameters)
    print output
    
if __name__ == '__main__':
    main()
