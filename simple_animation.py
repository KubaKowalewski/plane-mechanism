import numpy as np
from src.Slm import slm




slm_version = 2
if slm_version == 1:
    a,b,c = .1, .1, .275
if slm_version == 2:
    a,b,c = .44,.42,.47


SCALE = 1
A = np.array([a,a]) * SCALE
B = np.array([b,b,b,b]) * SCALE
C = np.array([c,c]) * SCALE
SLM = slm.mechanism(A,B,C,slm_version)
SLM.animate()





