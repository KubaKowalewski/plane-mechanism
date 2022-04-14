import numpy as np
from src.Slm import slm

# Set version and type and rhombous ratio
slm_version = 1
slm_group = "z"
slm_r = .9

# Specify link lengths for each version
if slm_version == 1:
    a,b,c = 0.1, 0.091,	0.29
if slm_version == 2:
    a,b,c = 0.1, .125, 0.2
if slm_version == 3:
    a,b,c =  0.11000, 0.49000, 0.43000

# Set the scale and create
SCALE = 1
A = np.array([a,a]) * SCALE
B = np.array([b,b,b,b]) * SCALE
C = np.array([c,c]) * SCALE
SLM = slm.mechanism(A,B,C,slm_version,slm_group,slm_r)

# Call animation function
SLM.update_state()
SLM.draw(debug=True)





