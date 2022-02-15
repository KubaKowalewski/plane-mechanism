from src.Slm import slm
import numpy as np

def optimize_SLM():
    # SLM Constants
    A = [1,1]
    B = [1,1,1,1]
    C = [2.75,2.75]
    theta_range = [-0.75,0.75]
    pass

def run(mode):
    if mode == "Default":
        # SLM Constants
        scale = 1
        A = np.array([1,1.2])*scale
        B = np.array([1,1,1,1])*scale
        C = np.array([2.65,2.75])*scale
        theta_range = [-0.75,0.75]
        # Create and animate
        SLM_1 = slm.mechanism(A,B,C,theta_range)
        SLM_1.animate(theta_range)
        
    elif mode == "Noise":
        scale = 10
        A = np.array([1,1])*scale
        B = np.array([1,1,1,1])*scale
        C = np.array([2.75,2.75])*scale
        theta_range = [-0.75,0.75]
        # Create and animate
        SLM_1 = slm.mechanism(A,B,C,theta_range,add_noise = True)
        SLM_1.animate(theta_range)


if __name__ == "__main__":
    mode = ["Default","Noise","Visualize Error"]
    run(mode[0])