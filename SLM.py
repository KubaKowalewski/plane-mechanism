import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats

class Mechanism:
    def __init__(self,A,B,C,theta_range,theta=0):
        # State Variables
        self.A = A
        self.B = B
        self.C = C
        self.links = A + B + C
        self.colors = ["red","red","blue","blue","blue","blue","purple","purple"] # Link colors
        self.theta_range = theta_range
        self.theta = theta
        self.update_state()
        # Plotting Variables
        self.fig, self.ax = plt.subplots()
        self.link_width = 4
        self.num_links = 8
        # Find path of SLM and linear fit
        self.path()
        
    def path(self):
        # Calculates path
        self.path_x = []
        self.path_y = []
        for i in np.arange(self.theta_range[0],self.theta_range[1],0.025):
            state_i = self.calculate_state(i)
            self.path_x.append(state_i[-1][0])
            self.path_y.append(state_i[-1][1])
        
        # Finds linear fit for path
        m, b, r, p, se = stats.linregress(self.path_y,self.path_x)
        self.fit_x = self.path_x
        print(m,b)
        linear = lambda x: m*x + b
        self.fit_y = list(map(linear,self.fit_x))
        self.r2 = r**2
        print(self.r2)
        print(self.fit_x)


    def norm(self,V):
        return sum(V*V)**0.5    

    def calculate_state(self,theta):
        N0 = np.array([0,0])
        N1 = np.array([self.A[0],0])
        N2 = N1 + np.array([self.A[1]*np.cos(theta),self.A[1]*np.sin(theta)])
        D2 = self.norm(N2)   
        theta_C1 = np.arctan(self.A[1]*np.sin(theta)/(self.A[0]+self.A[1]*np.cos(theta))) + np.arccos((D2**2+self.C[1]**2-self.B[1]**2)/(2*D2*self.C[1]))
        N3 = np.array([self.C[1]*np.cos(theta_C1),self.C[1]*np.sin(theta_C1)])
        theta_C0 = np.arctan(self.A[1]*np.sin(theta)/(self.A[0]+self.A[1]*np.cos(theta))) - np.arccos((D2**2+self.C[0]**2-self.B[0]**2)/(2*D2*self.C[0]))
        N4 = np.array([self.C[0]*np.cos(theta_C0),self.C[0]*np.sin(theta_C0)])
        D3_4 = self.norm(N4-N3)
        a = (self.B[3]**2-self.B[2]**2+D3_4**2)/(2*D3_4)
        h = np.sqrt(self.B[3]**2-a**2)
        P2 = N3 + a*(N4-N3)/D3_4
        N5x = P2[0]-h*(N4[1]-N3[1])/D3_4
        N5y = P2[1]+h*(N4[0]-N3[0])/D3_4
        N5 = [N5x,N5y]
        return([N0,N1,N2,N3,N4,N5])

    def update_state(self):
        self.N = self.calculate_state(self.theta)
    
    def draw(self):
        self.ax.cla()
        # draw path
        # plt.plot(self.path_x,self.path_y)
        plt.plot(self.fit_x,self.fit_y,'r')
        n_p = [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(0,3),(0,4)] # Node pairs
        # Draw all links
        for i in range(self.num_links):
            n0 = n_p[i][0]
            n1 = n_p[i][1]
            plt.plot([self.N[n0][0],self.N[n1][0]],[self.N[n0][1],self.N[n1][1]],linewidth=self.link_width,c=self.colors[i])
        # Draw all nodes
        for N in self.N:
            plt.scatter(N[0],N[1],s=50,c="black",zorder=10)
        plt.xlim([-.5, 3.75])
        plt.ylim([-1.75, 1.75])
        plt.gca().set_aspect("equal")
        plt.grid()
        return plt
        
    def update_frame(self,theta):
        self.theta = theta
        self.update_state()
        return self.draw()

    def animate(self,theta_range):
        self.slm_animation = animation.FuncAnimation(self.fig, self.update_frame, frames=np.append(np.arange(theta_range[0], theta_range[1], 0.025),np.arange(theta_range[1], theta_range[0], -0.025)), interval=100, repeat=True)
        plt.show()

