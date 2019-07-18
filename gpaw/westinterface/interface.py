
# TODO How to INIT input file?
# Ask Marco about:
# What input info does WEST need?
# What should the first potential be?
# How do we start up WEST/what executable should be used?
# Does WEST send a signal if it fails? If it is done?
# Can WEST read a signal to stop running if GPAWServer fails?

class WESTInterface:

    def __init__(self, calc_file, atoms=None, computer="niflheim", use_dummywest=False):
        from gpaw import GPAW
        calc = GPAW(calc_file, txt=None)
        self.calc = calc
        self.calc_file = calc_file
        if atoms is None:
            self.atoms = self.calc.get_atoms()
        else:
            self.atoms = atoms
        if self.atoms is None:
            raise ValueError("Atoms are required")
        self.use_dummywest = use_dummywest
        self.computer = computer

        from gpaw.westinterface import XMLReaderWriter
        self.xmlrw = XMLReaderWriter()

    def run(self, opts, dry_run=False):
        if self.computer.lower() == "gbar":
            cmd = self.gbar_submit(opts)
            submit_program = "qsub" 
        elif self.computer.lower() == "niflheim":
            cmd = self.niflheim_submit(opts)
            submit_program = "sbatch"
        else:
            raise ValueError("Computer '{}' not recognized".format(self.computer))

        self.write_initial_input(opts["Input"])
            
        if dry_run:
            return cmd
        else:
            import subprocess
            p = subprocess.Popen([submit_program], stdin=subprocess.PIPE)
            p.communicate(cmd.encode())
            return cmd


    def gbar_submit(self, opts):
        input_file = opts["Input"]
        output_file = opts["Output"]
        time = opts["Time"]
        gpaw_file = opts.get("Calcname") or self.calc_file
        jobname = opts.get("JobName") or "WEST-GPAW"
        ngpaw_nodes = int(opts["GPAWNodes"])
        nwest_nodes = int(opts["WESTNodes"])
        n_nodes = ngpaw_nodes + nwest_nodes
        ppn = 8
        from gpaw import westinterface
        interface_path = westinterface.__file__.replace("__init__.py", "")
        if self.use_dummywest:
            west_cmd = "gpaw-python " + interface_path + "westcodedummy.py"
        else:
            raise NotImplementedError


        script = ["#!/bin/sh",
                  "#PBS -q hpc",
                  "#PBS -N {}".format(jobname),
                  "#PBS -l nodes={}:ppn={}".format(n_nodes, ppn),
                  "#PBS -l walltime={}".format(time),
                  "cd $PBS_O_WORKDIR",
                  "OMP_NUM_THREADS=1 mpiexec -np {} gpaw-python {}gpawserver.py ./{} ./{} ./{} &".format(ngpaw_nodes*ppn, interface_path, input_file, output_file, gpaw_file),
                  "OMP_NUM_THREADS=1 mpiexec -np {} {} ./{} ./{}".format(nwest_nodes*ppn, west_cmd, input_file, output_file),
                  "wait",
                  "",
                  "exit $?"]
        return "\n".join(script)
                  
                  
    def niflheim_submit(self, opts):
        jobname = opts["JobName"]
        time = opts["Time"]
        input_file = opts["Input"]
        output_file = opts["Output"]
        gpaw_file = opts["Calcname"]
        ngpaw_nodes = int(opts["GPAWNodes"])
        nwest_nodes = int(opts["WESTNodes"])
        n_nodes = ngpaw_nodes + nwest_nodes
        partition = opts["Partition"]
        if partition.startswith("xeon8"):
            ppn = 8
        elif partition.startswith("xeon16"):
            ppn = 16
        elif partition.startswith("xeon24"):
            ppn = 24
        elif partition.startswith("xeon40"):
            ppn = 40
        else:
            raise ValueError("Partition '{}' not recognized".format(partition))

        from gpaw import westinterface
        interface_path = westinterface.__file__.replace("__init__.py", "")
        if self.use_dummywest:
            west_cmd = "gpaw-python " + interface_path + "westcodedummy.py"
        else:
            raise NotImplementedError

        script = ["#!/bin/bash",
                  "#SBATCH -J {}".format(jobname),
                  "#SBATCH --partition={}".format(partition),
                  "#SBATCH -N{}".format(n_nodes),
                  "#SBATCH -n{}".format(n_nodes*ppn),
                  "#SBATCH -t{}".format(time),
                  "cd $SLURM_SUBMIT_DIR",
                  "OMP_NUM_THREADS=1 mpiexec -np {} gpaw-python {}gpawserver.py ./{} ./{} ./{} &".format(ngpaw_nodes*ppn, interface_path, input_file, output_file, gpaw_file),
                  "OMP_NUM_THREADS=1 mpiexec -np {} {} ./{} ./{}".format(nwest_nodes*ppn, west_cmd, input_file, output_file),
                  "wait",
                  "",
                  "exit $?"]

        return "\n".join(script)



    def write_initial_input(self, filename):
        import numpy as np
        cell = self.atoms.get_cell()
        grid_shape = self.calc.wfs.gd.get_grid_point_coordinates().shape

        init_pot = np.random.rand(*grid_shape[1:])
        
        self.xmlrw.write(init_pot, cell, filename)
