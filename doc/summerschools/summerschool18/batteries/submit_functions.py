"""Submit a cell for calculation in the gbar. 

Example:

from submit_functions import submit
# 1 is the number of the cell with the calculation script.
# Put that at the end of the magic command _i => _i1
submit('name_of_calculation', _i1)
"""

import os
from subprocess import run, PIPE, STDOUT


submit_string = """#/bin/bash
#PBS -l nodes={nodes}:ppn=8
#PBS -N {name}
#PBS -l walltime={hours}:00:00

module load mpi/intel

cd {jobdir}
# Set a PYTHONPATH for the required packages if not already taken care of
export PYTHONPATH=/zhome/43/5/58576/gpaw-1.4.0-venv/lib/python3.6/site-packages:$PYTHONPATH
export PATH=/zhome/43/5/58576/gpaw-1.4.0-venv/bin:$PATH
export GPAW_SETUP_PATH=/zhome/43/5/58576/gpaw-1.4.0-venv/gpaw-setups-0.9.20000

mpirun gpaw-python calc.py
"""


def submit(name, txt, nodes=1, max_hours=1):
    # Make the new directory
    try:
        os.mkdir(name)
    except OSError:
        pass

    # Write the script using the input text
    script = open(os.path.join(name, 'calc.py'), 'w')
    script.write(txt)
    script.close()

    # Write the submit file
    submit_file = open(os.path.join(name, 'submit.sh'), 'w')
    cwd = os.getcwd()
    submit_file.write(submit_string.format(name=name,
                                           jobdir=os.path.join(cwd, name),
                                           nodes=nodes,
                                           hours='{0:02d}'.format(max_hours)))
    submit_file.close()

    # Change to the calculation folder
    os.chdir(os.path.expanduser(name))

    # Submit the calculation
    cp = run(['qsub submit.sh'],
             shell=True,
             stdin=PIPE,
             stdout=PIPE, stderr=STDOUT,
             universal_newlines=True)
    out = cp.stdout

    msg = 'The calculation "{0}" was submitted with the job ID: {1}'
    job_id = out.split('.')[0]
    print(msg.format(name, job_id))

    # Change back to original folder
    os.chdir(cwd)


def check_status(job_id):
    # Get the status
    cp = run(['qstat {0}'.format(job_id)],
             shell=True,
             stdin=PIPE,
             stdout=PIPE, stderr=STDOUT,
             universal_newlines=True)
    out = cp.stdout
    print(out)
