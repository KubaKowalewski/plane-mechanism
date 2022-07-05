import numpy as np
from src.Slm import slm

 # Class 3 Best
SLM = slm.mechanism(Rr=0.83448,Rf=False,A=0.1,Rw=-0.22,Rh=0.3)
SLM.colors = ["black","black","black","black","black","black","black","black"]
print(SLM.A,SLM.B,SLM.C)
SLM.draw(draw_path=False)





