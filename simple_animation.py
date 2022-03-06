import numpy as np
from src.Slm import slm

# SLM Constants
A = np.array([1,1])
B = np.array([1,1,1,1])
C = np.array([2.75,2.75])
SLM = slm.mechanism(A,B,C)
SLM.animate()




