from src.Slm import slm
import numpy as np

A = np.array([1,1])
B = np.array([1,1,1,1])
C = np.array([2.75,2.75])
SLM = slm.mechanism(A,B,C)
SLM.draw()
F = [0,-50000,0,0,0,0,0,0]
N = SLM.calculate_state(SLM.theta)
thetas = SLM.find_link_angles(N)
E = 200e9 
D = 5
SLM.path_stiffness(F,thetas,E,D)
# SLM.plot_stiffness()
SLM.animate_stiffness(save=False)



