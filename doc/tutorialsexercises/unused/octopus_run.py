#!/usr/bin/env python
from os import environ

label = 'oscillator'

octopus = environ['OCTOPUS_SCRIPT']
locals = {'label': label}
exec(open(octopus).read(), {}, locals)
exitcode = locals['exitcode']
if exitcode != 0:
    raise RuntimeError(('Octopus exited with exit code: %d.  ' +
                        'Check %s.log for more information.') %
                       (exitcode, label))
