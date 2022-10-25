from src.Slm import slm
import numpy as np

SLM = slm.mechanism(Rr=1,Rf=False,A=0.1,Rw=-0.5,Rh=0.3)
SLM.draw()

F = [0,-1000 ,0,0,0,0,0,0]
E = 200e9   
D = 1*10e-3
# SLM.draw()
SLM.path_stiffness(F,E,D)
# SLM.plot_stiffness()
SLM.animate_stiffness(path="./videos/stiffness_animation_v3y",save=False)






