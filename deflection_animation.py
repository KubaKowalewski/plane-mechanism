from src.Slm import slm
import numpy as np

# set slm version to analyze
slm_version = 2

if slm_version == 1:
    A = np.array([.1,.1])
    B = np.array([.1,.1,.1,.1])
    C = np.array([.275,.275])
if slm_version == 2:
    # a,b,c = 0.1,0.25,0.2 # weak in middle
    a,b,c = 0.1,0.3,0.2 # weak at edges
    A = np.array([a,a])
    B = np.array([b,b,b,b])
    C = np.array([c,c])

SLM = slm.mechanism(A,B,C,slm_version)
F = [0,-50000,0,0,0,0,0,0]
N = SLM.calculate_state(SLM.theta)
thetas = SLM.find_link_angles(N)
E = 200e9 
D = 5
SLM.path_stiffness(F,thetas,E,D)
SLM.plot_stiffness()
SLM.animate_stiffness(save=False)




