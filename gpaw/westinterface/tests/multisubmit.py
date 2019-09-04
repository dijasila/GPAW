import subprocess
import argparse

parser = argparse.ArgumentParser()

add = parser.add_argument

add("-t", "--time", type=str, default="02:00:00")
add("-N", "--name", type=str, default="SiH4")
add("-n", "--nodes", type=int, default=2)

args = parser.parse_args()

time = args.time
name = args.name
nodes = args.nodes



lines = [
    "#!/bin/sh",
    "#PBS -q hpc",
    "#PBS -N {}".format(name),
    "#PBS -l nodes={}:ppn=8".format(nodes),
    "#PBS -l walltime={}".format(time),
    "cd $PBS_O_WORKDIR",
    lambda s: "OMP_NUM_THREADS=1 mpiexec gpaw-python ./comparetester.py {}".format(s)]



#xc x, cutoff c, nbands n
script_args = [("-f LDA300_50", "-x LDA", "-c 300", "-n 50"), ("-f LDA400_50", "-x LDA", "-c 400", "-n 50"), ("-f LDA400_75", "-x LDA", "-c 400", "-n 75")]

script_args = [("-f PBEdef_50",)]

args = [[0], [0], [0], [4], [24], [0], script_args]


for ia, ags in enumerate(script_args):
    cmd_args = " ".join(ags)
    new_lines = lines.copy()
    new_lines[2] = new_lines[2] + "-" + str(ia)
    new_lines[-1] = new_lines[-1](cmd_args)
    script = "\n".join(new_lines) + "\n"
    p = subprocess.Popen(["qsub"], stdin=subprocess.PIPE)
    p.communicate(script.encode())
    print(script)
    print("")
