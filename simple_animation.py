import numpy as np
from src.Slm import slm

# SLM Constants
A = np.array([1,1.1])
B = np.array([1,1,1.2,1])
C = np.array([2.65,2.75])
MAX_THETA = 0.6
theta_range = [-MAX_THETA,MAX_THETA]
SLM = slm.mechanism(A,B,C,theta_range,add_noise = True)
SLM.animate(theta_range)
