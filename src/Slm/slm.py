import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import random

class mechanism:
    # Initializes mechanism
    def __init__(self,A,B,C,theta_range,theta=0,**kwargs):
        # State Variables
        self.A = A
        self.B = B
        self.C = C
        self.colors = ["red","red","blue","blue","blue","blue","purple","purple"] # Link colors
        self.theta_range = theta_range
        self.theta = theta
        self.link_theta = [] # list of all link angles in mechanism
        if "add_noise" in kwargs and "noise" in kwargs and kwargs["add_noise"]:
            max_noise = kwargs["noise"]
            noise = lambda x: x + random.uniform(-max_noise,max_noise)
            self.A = list(map(noise,A))
            self.B = list(map(noise,B))
            self.C = list(map(noise,C))
        # Check if this is valid SLM
        if not self.is_valid():
            raise Exception("Invalid SLM")
        print("Valid!")
        # Store metrics for measuring model performance
        self.coef_list = []
        self.rmse_list = []
        self.update_state()
        # Plotting Variables
        self.fig, self.ax = plt.subplots()
        # plt.close()
        self.link_width = 4
        self.num_links = 8
        # Find path of SLM and linear fit
        self.path(save_error=False)
    
    # Returns angle between L2 and L3
    def law_of_cos(self,L1,L2,L3):
        return np.arccos((L2**2+L3**2-L1**2)/(2*L2*L3))

    # Checks if link lengths create valid mechanism
    def is_valid(self):
        valid_lower = (self.A[0] + self.A[1] + self.B[0] > self.C[0]) and (self.A[0] + self.A[1] + self.B[1] > self.C[1])
        theta0 = (np.pi/2)-self.law_of_cos(self.B[0],self.C[0],self.A[0]+self.A[1])
        theta1 = (np.pi/2)-self.law_of_cos(self.B[1],self.C[1],self.A[0]+self.A[1])
        valid_upper = (self.B[2] > self.C[0]*np.cos(theta0))and (self.B[3] > self.C[1]*np.cos(theta1))
        return valid_lower and valid_upper

    # Updates mechanism with new links
    def update_links(self,A=[],B=[],C=[],**kwargs):
        # Method 1 of passing new data in
        if len(A) > 0: self.A = A
        if len(B) > 0: self.B = B
        if len(C) > 0: self.C = C
        # Method 2 of passing new data in
        if "dA" in kwargs: self.A += kwargs["dA"]
        if "dB" in kwargs: self.B += kwargs["dB"]
        if "dC" in kwargs: self.C += kwargs["dC"]

    # Swaps links in the symmetric groups
    def swap_symmetric(self):
        self.A = np.array([self.A[1],self.A[0]])
        # self.B = np.array([self.B[2],self.B[3],self.B[0],self.B[1]])
        
    def swap_anti_symmetric(self):
        pass

    # Calculates path of mechanism across theta range        
    def path(self,save_error=True,save_path=False):
        # Calculates path
        self.path_x = []
        self.path_y = []
        step_size = 0.025
        for i in np.arange(self.theta_range[0],self.theta_range[1]+step_size,step_size):
            state_i = self.calculate_state(i)
            self.path_y.append(state_i[-1][0])
            self.path_x.append(state_i[-1][1])
        # Finds linear fit for path
        self.path_x_T = np.array(self.path_x).reshape(-1,1) # Transpose
        model = LinearRegression().fit(self.path_x_T, self.path_y)
        self.fit_x = self.path_x
        self.fit_y = model.predict(self.path_x_T)
        self.coef = model.coef_
        self.rmse = mean_squared_error(self.path_y,self.fit_y,squared=False)
        self.path_list = list(zip(self.path_x,self.path_y))
        return(self.path_x,self.path_y)
    
    # Saves current error
    def save_error(self):
        self.coef_list.append(self.coef)
        self.rmse_list.append(self.rmse)
    
    def nodes_to_theta(A,B):
        pass

    # Finds current state defined by nodes of mechanism  
    def calculate_state(self,theta):
        norm = lambda V: sum(V*V)**0.5
        N0 = np.array([0,0])
        N1 = np.array([self.A[0],0])
        N2 = N1 + np.array([self.A[1]*np.cos(theta),self.A[1]*np.sin(theta)])
        D2 = norm(N2)   
        theta_C1 = np.arctan(self.A[1]*np.sin(theta)/(self.A[0]+self.A[1]*np.cos(theta))) + np.arccos((D2**2+self.C[1]**2-self.B[1]**2)/(2*D2*self.C[1]))
        N3 = np.array([self.C[1]*np.cos(theta_C1),self.C[1]*np.sin(theta_C1)])
        theta_C0 = np.arctan(self.A[1]*np.sin(theta)/(self.A[0]+self.A[1]*np.cos(theta))) - np.arccos((D2**2+self.C[0]**2-self.B[0]**2)/(2*D2*self.C[0]))
        N4 = np.array([self.C[0]*np.cos(theta_C0),self.C[0]*np.sin(theta_C0)])
        D3_4 = norm(N4-N3)
        a = (self.B[3]**2-self.B[2]**2+D3_4**2)/(2*D3_4)
        h = np.sqrt(self.B[3]**2-a**2)
        P2 = N3 + a*(N4-N3)/D3_4
        N5x = P2[0]-h*(N4[1]-N3[1])/D3_4
        N5y = P2[1]+h*(N4[0]-N3[0])/D3_4
        N5 = [N5x,N5y]
        return([N0,N1,N2,N3,N4,N5])

    # Updates all nodes within mechanism
    def update_state(self):
        self.N = self.calculate_state(self.theta)

    # Plot path and fit of mechanism
    def plot_path(self,title="Mechanism Path"):
        self.ax.cla()
        plt.plot(self.path_x,self.path_y)
        plt.plot(self.fit_x,self.fit_y,'r')
        plt.legend(["Path","Linear Fit"])
        plt.title(title)
        plt.grid()
        plt.show()

    # Plots error of mechanism
    def plot_error(self,**kwargs):
        label = "Iterations"
        x_values = range(len(self.coef_list))
        # Check for alternative label
        if len(kwargs) != 0:
            label = list(kwargs.keys())[0]
            x_values = kwargs[label]
        # Resize figure
        self.ax.cla()
        fig = plt.gcf()
        fig.set_size_inches(10, 12)
        # Plot Convergence of MSE
        plt.subplot(2,1,1)
        plt.plot(x_values,self.rmse_list)
        plt.title("RMSE vs "+label)
        plt.xlabel(label)
        plt.ylabel("RMSE")
        plt.grid()
        # Plot convergence of Slope
        plt.subplot(2,1,2)
        plt.plot(x_values,self.coef_list)
        plt.title("Slope vs "+label)
        plt.xlabel(label)
        plt.ylabel("Slope")
        plt.grid()
        # Display plot
        plt.show()

    # Draws mechanism and path in for its current state
    def draw(self):
        self.ax.cla()
        # draw path
        plt.plot(self.path_y,self.path_x)
        plt.plot(self.fit_y,self.fit_x,'r')
        n_p = [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(0,3),(0,4)] # Node pairs
        # Draw all links
        for i in range(self.num_links):
            n0 = n_p[i][0]
            n1 = n_p[i][1]
            plt.plot([self.N[n0][0],self.N[n1][0]],[self.N[n0][1],self.N[n1][1]],linewidth=self.link_width,c=self.colors[i])
        # Draw all nodes
        for N in self.N:
            plt.scatter(N[0],N[1],s=50,c="black",zorder=10)
        plt.xlim([-.5, self.A[0]+self.A[1]+1.5*max(self.B)])
        plt.ylim([-1.5*max(self.B), 1.5*max(self.B)])
        plt.gca().set_aspect("equal")
        plt.grid()
        return plt
    
    # Used to update frames ion animation
    def update_frame(self,theta):
        self.theta = theta
        self.update_state()
        return self.draw()

    # Animates mechanism moving across theta range
    def animate(self,theta_range):
        print("ok")
        print(theta_range)
        self.slm_animation = animation.FuncAnimation(self.fig, self.update_frame, frames=np.append(np.arange(theta_range[0], theta_range[1], 0.025),np.arange(theta_range[1], theta_range[0], -0.025)), interval=100, repeat=True)
        plt.show()

