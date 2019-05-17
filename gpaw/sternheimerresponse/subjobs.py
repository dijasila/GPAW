import numpy as np
#import subprocess
import os



cutoffs = np.arange(100, 1000, 50)

setups = ["ae", "paw", "only1s"]


for setup in setups:
    for cutoff in cutoffs:
        cmd = ["mq", "submit", "./sternheimer.py+H_" + str(int(cutoff)) + "_"  + setup, "-R", "1:10m"]
        cmd = " ".join(cmd)
        os.system(cmd)
        #subprocess.run(cmd)
