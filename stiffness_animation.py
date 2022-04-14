from src.Slm import slm
import numpy as np

# set slm version to analyze
slm_version = 1

if slm_version == 1:
    slm_group = "z"
    ratio = 0.9
    a,b,c = 0.1, 0.091,	0.29
    A = np.array([a,a])
    B = np.array([b,b,b,b])
    C = np.array([c,c])

if slm_version == 2:
    slm_group = "z"
    ratio = 0.9
    a,b,c = 0.1, 0.091,	0.29
    A = np.array([a,a])
    B = np.array([b,b,b,b])
    C = np.array([c,c])

if slm_version == 3:
    slm_group = "y"
    ratio = 0.5
    a,b,c = 0.11000, 0.49000, 0.43000
    A = np.array([a,a])
    B = np.array([b,b,b,b])
    C = np.array([c,c])

SLM = slm.mechanism(A,B,C,slm_version,slm_group,ratio)
F = [0,-100 ,0,0,0,0,0,0]
N = SLM.calculate_state(SLM.theta)
thetas = SLM.find_link_angles(N)
E = 200e9
D = 5
# SLM.draw()
SLM.path_stiffness(F,thetas,E,D)
# SLM.plot_stiffness()
SLM.animate_stiffness(path="./videos/stiffness_animation_v3y",save=False)





