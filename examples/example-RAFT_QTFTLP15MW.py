# example script for running RAFT with second-order loads computed internally with the slender-body approximation based on Rainey's equation
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft

import sys

# open the design YAML file and parse it into a dictionary for passing to raft
# flNm = r"X:\00002 - Mocean employees\Mats\RAFT\examples\TLP15MW-RAFT_QTF"
# with open(flNm + '.yaml') as file:
#    design = yaml.load(file, Loader=yaml.FullLoader)

yaml_path = sys.argv[1]
with open(yaml_path) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)
output_dir = os.path.dirname(flNm)
# Create the RAFT model (will set up all model objects based on the design dict)
model = raft.Model(design)
model.output_dir = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Orcaflex validatie\15MW\FinalRAFTdata"  # <==== ADD THIS LINE
# Evaluate the system properties and equilibrium position before loads are applied
model.analyzeUnloadedflex(ballast=1)

# Compute natural frequencie
model.solveEigenFlex(display=0)

# Due to the linearization of the quadratic drag term in RAFT, the QTFs depend on the sea state specified in the input file.
# If more than one case is analyzed, the outputs are numbered sequentially.
# Two output files are generated:
# - The QTF, following WAMIT .12d file format. File name is qtf-slender_body-total_Head#p##_Case#_WT#.12d
# - The RAOs used to computed the QTFs, following WAMIT .4 file format. File name is qtf-slender_body-total_Head#p##_Case#_WT#
# The Head#p## in the file name indicates the wave heading in degrees (p replaces the decimal point). 
# Case number starts at 1, but turbine at 0 in conformity with the rest of the code.
#model.analyzeCases(display=1)
model.analyzeCasescompflex(display=1)
#model.analyzeCasescompflex(display=1)
#model.solveEigenFlex(display=1)

#model.plotResponsesflex()



# Visualize the system in its most recently evaluated mean offset position
#model.plot()

#plt.show()

# 0.02
# 12.37