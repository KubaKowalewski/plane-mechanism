import SLM

def run(mode):
    if mode == "Default":
        # SLM Constants
        A = [1,1]
        B = [1,1.2,1,1.2]
        C = [2.656889,2.738441]
        theta_range = [-0.75,0.75]
    elif mode == "Noise":
        import random
        max_noise = 0.1
        A = [1,1]
        B = [1,1,1,1]
        C = [2.75,2.75]
        theta_range = [-0.75,0.75]
        noise = lambda x: x + random.uniform(-max_noise,max_noise)
        A = list(map(noise,A))
        B = list(map(noise,B))
        C = list(map(noise,C))
    # Create and animate
    SLM_1 = SLM.Mechanism(A,B,C,theta_range)
    SLM_1.animate(theta_range)

if __name__ == "__main__":
    modes = ["Default","Noise"]
    run(modes[1])