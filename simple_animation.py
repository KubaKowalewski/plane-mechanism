import numpy as np
from src.Slm import slm

# SLM Constants
SCALE = 6
# a, b, c = .1, .28, .35 # Poor max stiffness
a, b, c = .1, .29, .42 # Good max stiffness

A = np.array([a,a]) * SCALE
B = np.array([b,b,b,b]) * SCALE
C = np.array([c,c]) * SCALE
SLM = slm.mechanism(A,B,C)
SLM.draw()





