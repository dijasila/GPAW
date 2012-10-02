from gpaw.parameters import InputParameters
from gpaw.solvation.contributions import CONTRIBUTIONS

# contribution -> mode -> parameter -> default
PARAMETERS = dict(
    [(cname, dict(
        [(None, {})] + \
        [(mname, mode.default_parameters) for mname, mode in modes.iteritems()]
        )
      ) for cname, modes in CONTRIBUTIONS.iteritems()]
    )


class SolvationInputParameters(InputParameters):
    def __init__(self, **kwargs):
        InputParameters.__init__(self, **kwargs)
        if not 'solvation' in self:
            self['solvation'] = dict(
                [(k, {'mode':None}) for k in PARAMETERS]
                )

    def read(self, reader):
        InputParameters.read(self, reader)
        # XXX TODO: implement io for custom PoissonSolvers
        self.poissonsolver = None
        try:
            solvation = reader['SolvationParameters']
        except KeyError:
            solvation = {}
        update_parameters(self.solvation, solvation)


def update_parameters(old, update):
    """
    updates solvation parameters old from update

    returns modified keys as list of tuples (contribution_key, parameter_key)

    If mode has changed, default parameters for this mode are read
    and then updated. Only mode itself and changes during this update are
    then reported as modified.

    raises KeyError, if a parameter not defined in PARAMETERS
    is updated

    raises ValueError, if mode is set to a mode not defined in
    PARAMETERS
    """
    modified = []
    for contrib, cparams in update.iteritems():
        if contrib not in PARAMETERS:
            raise KeyError(
                'Unknown solvation contribution: %s' % (contrib, )
                )
        if 'mode' in cparams and cparams['mode'] != old[contrib]['mode']:
            modified.append((contrib, 'mode'))
            if cparams['mode'] not in PARAMETERS[contrib]:
                raise ValueError(
                    'Unknown mode "%s" for contribution "%s".' \
                    % (cparams['mode'], contrib)
                    )
            old[contrib] = PARAMETERS[contrib][cparams['mode']].copy()
            old[contrib]['mode'] = cparams['mode']
            cparams = cparams.copy()
            del cparams['mode']
        for pkey, pvalue in cparams.iteritems():
            if pkey not in old[contrib]:
                raise KeyError(
                    'Unknown solvation parameter "%s" '
                    'for contribution "%s" in mode "%s".' \
                    % (pkey, contrib, old[contrib]['mode'])
                    )
            if pvalue != old[contrib][pkey]:
                old[contrib][pkey] = pvalue
                modified.append((contrib, pkey))
    return modified
