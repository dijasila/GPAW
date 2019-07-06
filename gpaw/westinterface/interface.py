


class WESTInterface:

    def __init__(self, calc, atoms=None, computer="niflheim", use_dummywest=False):
        if calc is None:
            raise ValueError("Calculator is required")
        self.calc = calc
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
        elif self.computer.lower() == "niflheim":
            cmd = self.niflheim_submit(opts)
        else:
            raise ValueError("Computer '{}' not recognized".format(self.computer))

        self.write_initial_input(opts["Input"])
            
        if dry_run:
            return cmd
        else:
            raise NotImplementedError


    def gbar_submit(self, opts):
        input_file = opts["Input"]
        output_file = opts["Output"]
        gpaw_file = opts["Calcname"]
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
                  "#PBS -N {}".format(opts["JobName"]),
                  "#PBS -l nodes={}:ppn={}".format(n_nodes, ppn),
                  "#PBS -l walltime={}".format(opts["Time"]),
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
