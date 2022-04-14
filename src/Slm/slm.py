import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import gridspec
import random

class mechanism:
    # Initializes mechanism
    def __init__(self,A,B,C,version=1,group="x",rhombus_ratio=1,**kwargs):
        # State Variables
        self.version = version # specifies which SLM version is being used
        self.group = group
        self.A = A
        self.B = B
        self.C = C
        self.rhombus_ratio = rhombus_ratio
        self.adjust_rhombus()
        self.lengths = np.array([self.A[1],self.B[1],self.B[0],self.B[3],self.B[2],self.C[1],self.C[0]])
        self.colors = ["red","red","blue","blue","blue","blue","purple","purple"] # Link colors
        # Sets the animation sampling resolution
        self.resolution = 2
        self.step_size = 10**(-self.resolution)
        self.theta_range = self.calculate_range()
        self.theta = 0
        # Adds noise to links in mechanism
        if "add_noise" in kwargs and "noise" in kwargs and kwargs["add_noise"]:
            max_noise = kwargs["noise"]
            noise = lambda x: x + random.uniform(-max_noise,max_noise)
            self.A = list(map(noise,A))
            self.B = list(map(noise,B))
            self.C = list(map(noise,C))
        # Check if this is valid SLM
        self.is_valid()
        # Store metrics for measuring model performance
        self.coef_list = []
        self.rmse_list = []
        self.update_state()
        self.link_width = 4
        self.num_links = 8
        # Find path of SLM and linear fit
        self.path(save_error=False)

    # Returns angle between L2 and L3
    def law_of_cos(self,L1,L2,L3):
        return np.arccos((L2**2+L3**2-L1**2)/(2*L2*L3))
    
    # Adjusts the B1,B3,C1 links for PL mechanisms of Y,Z group based on rhombus_ratio
    def adjust_rhombus(self):
        if self.group == "y" or self.group == "z":
            if self.version == 1 or self.version == 2:
                theta_c1 = self.law_of_cos(self.B[1],self.C[1],self.A[0]+self.A[0])
                y0 = self.C[1]*np.sin(theta_c1)
                y1 = y0*self.rhombus_ratio
                Cx = self.C[1]*np.cos(theta_c1)
                Bx = abs(Cx-self.A[0]-self.A[1])
                C_new = np.sqrt(y1**2+Cx**2)
                B_new = np.sqrt(y1**2+Bx**2)
                self.C[1] = C_new
                self.B[1] = B_new
                self.B[3] = B_new
            if self.version == 3:
                theta_b1 = self.law_of_cos(self.C[1],self.B[1],self.A[0]+self.A[0])
                y0 = self.B[1]*np.sin(theta_b1)
                y1 = y0*self.rhombus_ratio
                Bx = self.B[1]*np.cos(theta_b1)
                Cx = abs(Bx-self.A[0]-self.A[1])
                C_new = np.sqrt(y1**2+Cx**2)
                B_new = np.sqrt(y1**2+Bx**2)
                self.C[1] = C_new
                self.B[1] = B_new
                self.B[3] = B_new

    ####################
    # Update Functions
    ####################

    # Updates mechanism with new links
    def update_links(self,A=[],B=[],C=[],group = "x",rr=1,**kwargs):
        # Method 1 of passing new data in
        if len(A) > 0: self.A = A
        if len(B) > 0: self.B = B
        if len(C) > 0: self.C = C
        self.group = group
        self.rhombus_ratio = rr
        self.adjust_rhombus()
        # Method 2 of passing new data in
        if "dA" in kwargs: self.A += kwargs["dA"]
        if "dB" in kwargs: self.B += kwargs["dB"]
        if "dC" in kwargs: self.C += kwargs["dC"]
        # Check if valid
        self.is_valid()
        # Find range
        self.theta_range = self.calculate_range()

    # Swaps links in the symmetric groups
    def swap_symmetric(self):
        self.A = np.array([self.A[1],self.A[0]])
        # self.B = np.array([self.B[2],self.B[3],self.B[0],self.B[1]])
        
    def swap_anti_symmetric(self):
        pass
    
    ####################
    # Forward Kinematics
    ####################

    # Checks if link lengths create valid mechanism
    def is_valid(self):
        # Checks if valid link lengths are provided for specific mechanism version
        if self.version == 1:
            valid_B_C = min(self.B) < min(self.C)
            valid_obtuse_upper = (np.sqrt((self.A[0]+self.A[1])**2+self.B[1]**2) < self.C[1]) and (self.B[1] + self.A[0] + self.A[1] > self.C[1])
            valid_obtuse_lower = (np.sqrt((self.A[0]+self.A[1])**2+self.B[0]**2) < self.C[0]) and (self.B[0] + self.A[0] + self.A[1] > self.C[0])
            # Check that upper links can connect to form an end point
            theta0 = (np.pi/2)-self.law_of_cos(self.B[0],self.C[0],self.A[0]+self.A[1])
            theta1 = (np.pi/2)-self.law_of_cos(self.B[1],self.C[1],self.A[0]+self.A[1])
            valid_end_point = (self.B[2] > self.C[0]*np.cos(theta0)) and (self.B[3] > self.C[1]*np.cos(theta1))
            valid =  valid_B_C and valid_obtuse_upper and valid_obtuse_lower and valid_end_point
            # continous_range = (np.sqrt(((self.A[0] + self.A[1])**2) + self.B[0]**2) < self.C[0]) and (np.sqrt(((self.A[0] + self.A[1])**2) + self.B[1]**2) < self.C[1])
            # valid_lower = (self.A[0] + self.A[1] + self.B[0] > self.C[0]) and (self.A[0] + self.A[1] + self.B[1] > self.C[1])
            # theta0 = (np.pi/2)-self.law_of_cos(self.B[0],self.C[0],self.A[0]+self.A[1])
            # theta1 = (np.pi/2)-self.law_of_cos(self.B[1],self.C[1],self.A[0]+self.A[1])
            # valid_upper = (self.B[2] > self.C[0]*np.cos(theta0)) and (self.B[3] > self.C[1]*np.cos(theta1))
        if self.version == 2:
            valid_B_C = min(self.B) < min(self.C)
            valid_acute_upper = (np.sqrt((self.A[0]+self.A[1])**2+self.B[1]**2) > self.C[1]) and (self.C[1] + self.B[1] > self.A[0] + self.A[1])
            valid_acute_lower = (np.sqrt((self.A[0]+self.A[1])**2+self.B[0]**2) > self.C[0]) and (self.C[0] + self.B[0] > self.A[0] + self.A[1])
            valid = valid_B_C and valid_acute_upper and valid_acute_lower
            # Check that upper links can connect to form an end point
            theta0 = (np.pi/2)-self.law_of_cos(self.B[0],self.C[0],self.A[0]+self.A[1])
            theta1 = (np.pi/2)-self.law_of_cos(self.B[1],self.C[1],self.A[0]+self.A[1])
            valid_end_point = (self.B[2] > self.C[0]*np.cos(theta0)) and (self.B[3] > self.C[1]*np.cos(theta1))
            valid =  valid_B_C and valid_acute_upper and valid_acute_lower and valid_end_point
        if self.version == 3:
            valid_B_C = min(self.B) > min(self.C)
            valid_acute_upper = (np.sqrt((self.A[0]+self.A[1])**2+self.C[1]**2) > self.B[1]) and (self.C[1] + self.B[1] > self.A[0] + self.A[1])
            valid_acute_lower = (np.sqrt((self.A[0]+self.A[1])**2+self.C[0]**2) > self.B[0]) and (self.C[0] + self.B[0] > self.A[0] + self.A[1])
            valid_obtuse_upper = (np.sqrt((self.A[0]+self.A[1])**2+self.C[1]**2) < self.B[1]) and (self.C[1] + self.A[0] + self.A[1] > self.B[1])
            valid_obtuse_lower = (np.sqrt((self.A[0]+self.A[1])**2+self.C[0]**2) < self.B[0]) and (self.C[0] + self.A[0] + self.A[1] > self.B[0])
            # Check that upper links can connect to form an end point
            theta0 = (np.pi/2)-self.law_of_cos(self.C[0],self.B[0],self.A[0]+self.A[1])
            theta1 = (np.pi/2)-self.law_of_cos(self.C[1],self.B[1],self.A[0]+self.A[1])
            valid_end_point = (self.B[2] > self.B[0]*np.cos(theta0)) and (self.B[3] > self.B[1]*np.cos(theta1))
            valid = ((valid_acute_upper and valid_acute_lower) or (valid_obtuse_upper and valid_obtuse_lower)) and valid_B_C and valid_end_point
        if not valid:
            raise Exception("Invalid SLM")
        else:
            print("Valid SLM!")

    # Find workspace limits defined by maximum theta value
    def calculate_range(self):
        limit = 0.90 # Per side so 90% total
        a = np.average(self.A)
        b_lower = min([self.B[0],self.B[1]])
        c = min(self.C)
        if self.version == 1 or self.version == 3:
            self.MAX_THETA = 0.98*np.pi-self.law_of_cos(c-b_lower,a,a)
        if self.version == 2:
            self.MAX_THETA = 0.98*self.law_of_cos(b_lower,c-a,a)
        self.MAX_THETA = round(self.MAX_THETA,self.resolution)
        N = self.calculate_state(self.MAX_THETA)
        self.MAX_L = abs(2*N[5][1])
        self.LIMIT_THETA = round(self.MAX_THETA*limit,self.resolution)
        N = self.calculate_state(self.LIMIT_THETA)
        self.LIMIT_L = abs(2*N[5][1])
        return([-self.LIMIT_THETA,self.LIMIT_THETA])
    
    # Forward kinematics for each node in mechanism without assuming equal lengths in respective link groups
    # This allows any valid PL and PL-Like Mechanism to be calclated  
    def calculate_state(self,theta):
        norm = lambda V: sum(V*V)**0.5
        N0 = np.array([0,0])
        if self.version == 1 or self.version == 2:
            N1 = np.array([self.A[0],0])
            N2 = N1 + np.array([self.A[1]*np.cos(theta),self.A[1]*np.sin(theta)])
            D2 = norm(N2)   
            theta_C1 = np.arctan(self.A[1]*np.sin(theta)/(self.A[0]+self.A[1]*np.cos(theta))) + np.arccos((D2**2+self.C[1]**2-self.B[1]**2)/(2*D2*self.C[1]))
            N3 = np.array([self.C[1]*np.cos(theta_C1),self.C[1]*np.sin(theta_C1)])
            theta_C0 = np.arctan(self.A[1]*np.sin(theta)/(self.A[0]+self.A[1]*np.cos(theta))) - np.arccos((D2**2+self.C[0]**2-self.B[0]**2)/(2*D2*self.C[0]))
            N4 = np.array([self.C[0]*np.cos(theta_C0),self.C[0]*np.sin(theta_C0)])
        if self.version == 3:
            N1 = np.array([-self.A[0],0])
            N2 = N1 + np.array([-self.A[1]*np.cos(theta),-self.A[1]*np.sin(theta)])
            D2 = norm(N2)  
            theta_C1 = self.law_of_cos(self.B[1],self.C[1],D2)-theta/2
            N3 = np.array([-self.C[1]*np.cos(theta_C1),self.C[1]*np.sin(theta_C1)])
            theta_C0 = self.law_of_cos(self.B[0],self.C[0],D2)+theta/2
            N4 = np.array([-self.C[0]*np.cos(theta_C0),-self.C[0]*np.sin(theta_C0)])
        # End point is at intersection of the two circles defined by links B2, B2 located at N3 and N4
        # https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect      
        D3_4 = norm(N4-N3)
        a = (self.B[3]**2-self.B[2]**2+D3_4**2)/(2*D3_4)
        h = np.sqrt(self.B[3]**2-a**2)
        P2 = N3 + a*(N4-N3)/D3_4
        # Outer intersection of circles
        if self.version == 1 or self.version == 3:
            N5y = P2[1]+h*(N4[0]-N3[0])/D3_4
            N5x = P2[0]-h*(N4[1]-N3[1])/D3_4
        # Inner intersection of circles
        if self.version == 2:
            N5y = P2[1]-h*(N4[0]-N3[0])/D3_4
            N5x = P2[0]+h*(N4[1]-N3[1])/D3_4
        N5 = np.array([N5x,N5y])
        # Apply z group reflection
        if self.group == "z":
            b = N5y/N5x
            N3x = (N3[0]*(1-b**2)-2*b*(-N3[1]))/(1+b**2)
            N3y = (N3[1]*(b**2-1)+2*(b*N3[0]))/(1+b**2)
            N3 = np.array([N3x,N3y])
        return([N0,N1,N2,N3,N4,N5])
    
    # Updates all nodes within mechanism
    def update_state(self):
        self.N = self.calculate_state(self.theta)

    ####################
    # Path Calculations
    ####################

    # Saves current error
    def save_error(self):
        self.coef_list.append(self.coef)
        self.rmse_list.append(self.rmse)

    # Finds the angle a pair of nodes makes with global coordinate frame
    def n2t(self,A,B):
        x = B[0]-A[0]
        y = B[1]-A[1]
        return np.arctan2(y,x)
    
    def find_link_angles(self,N):
        return([self.n2t(N[1],N[2]),self.n2t(N[2],N[3]),self.n2t(N[2],N[4]),self.n2t(N[3],N[5]),self.n2t(N[4],N[5]),self.n2t(N[0],N[3]),self.n2t(N[0],N[4])])

    # Calculates path of mechanism across theta range        
    def path(self,save_error=True):
        # Calculates path
        self.path_x = []
        self.path_y = []
        for i in np.arange(self.theta_range[0],self.theta_range[1]+self.step_size,self.step_size):
            N = self.calculate_state(i)
            self.path_y.append(N[-1][0])
            self.path_x.append(N[-1][1])

        # Finds linear fit for path
        self.path_x_T = np.array(self.path_x).reshape(-1,1) # Transpose
        model = LinearRegression().fit(self.path_x_T, self.path_y)
        self.fit_x = self.path_x
        self.fit_y = model.predict(self.path_x_T)
        self.coef = model.coef_
        self.rmse = mean_squared_error(self.path_y,self.fit_y,squared=False)
        self.path_list = list(zip(self.path_x,self.path_y))
        return(self.path_x,self.path_y)

    ####################
    # Stiffness Calculations
    ####################

    # Returns stiffness in Global y direction of mechanism for a given external force vector F (N) 
    # Inputs -> : thetas (array:rad) link angle relative to global coordinate frame
        # parameters: Elastic modulus E (Pa), Link lengths (mm)
    # Outputs -> : k_y - stiffness in global y direction, d - displacment vector for all nodes in mechanism
    def calculate_stiffness(self,F,thetas,E,D):
        # Code numbers for each link found externally
        CN = [[9,10,7,8],[7,8,5,6],[7,8,3,4],[5,6,1,2],[3,4,1,2],[11,12,5,6],[11,12,3,4]]
        NDOF = 8 # SLM has 8 dof structurally
        A = (np.pi/4)*(D/1000)**2
        K_members = []
        # Finds the stiffness matrix of each link in global coordinate frame
        for i in range(len(thetas)):
            T = thetas[i]
            L = self.lengths[i]
            K = np.array([[np.cos(T)**2,np.cos(T)*np.sin(T),-np.cos(T)**2,-np.cos(T)*np.sin(T)],
                        [np.cos(T)*np.sin(T),np.sin(T)**2,-np.cos(T)*np.sin(T),-np.sin(T)**2],
                        [-np.cos(T)**2,-np.cos(T)*np.sin(T),np.cos(T)**2,np.cos(T)*np.sin(T)],
                        [-np.cos(T)*np.sin(T),-np.sin(T)**2,np.cos(T)*np.sin(T),np.sin(T)**2]])
            K = (K*E*A)/L
            K_members.append(K)
        # Construct global stiffness matrix
        Ks = np.zeros((NDOF,NDOF))
        for m in range(len(CN)):
            cn_m = CN[m]
            for i in range(len(cn_m)):
                row = cn_m[i]
                if row <= NDOF:
                    for j in range(len(cn_m)):
                        col = cn_m[j]
                        if col <= NDOF:
                            Ks[row-1][col-1]+=K_members[m][i][j]
        tol = 0.0001
        # Use psudeo inverse to find best fit for compliance matrix
        Cs = np.linalg.pinv(Ks, rcond=tol)
        k_y = 1/(Cs[1][1])
        d = Cs@F
        return(k_y,d)

    # Finds node location of deformed structure
    def find_deformed_nodes(self,d,N):
        DOF_2_Node = [(5,1),(5,0),(4,1),(4,0),(3,1),(3,0),(2,1),(2,0)]
        dN = np.copy(N)
        for i in range(len(DOF_2_Node)):
            DOF = DOF_2_Node[i]
            dN[DOF[0]][DOF[1]] += d[i]
        return dN
    
    # Finds stiffness of mechanism along entire path
    def path_stiffness(self,F,thetas,E,D):
        self.ky_array = []
        self.d_array = []
        self.k_path_x = []
        self.k_path_y = []
        self.theta_array = np.linspace(self.theta_range[0],self.theta_range[1],
                            (self.theta_range[1]-self.theta_range[0])/self.step_size)
        for i in self.theta_array:
            N = self.calculate_state(i)
            thetas = self.find_link_angles(N)
            k_res = self.calculate_stiffness(F,thetas,E,D)
            ky = k_res[0]
            d = k_res[1]
            dN = self.find_deformed_nodes(d,N)
            self.ky_array.append(ky)
            self.d_array.append(d)
            self.k_path_y.append(dN[-1][0])
            self.k_path_x.append(dN[-1][1])
        return self.ky_array
    

    # Returns all relevant system properties
    def find_system_properties(self,F,E,D,P):
        n = 100
        thetas = np.linspace(-self.LIMIT_THETA,self.LIMIT_THETA,n)
        # Maximum stiffness
        full_stiffness = self.path_stiffness(F,thetas,E,D)
        k_max = max(full_stiffness) # Max stiffness
        # Minimum stiffness
        k_min = min(full_stiffness) # min stiffness
        L = np.sum(np.concatenate((self.A,self.B,self.C)))
        A = (np.pi/4)*(D/1000)**2
        weight = L*A*P
        return(k_max,k_min,self.LIMIT_L,weight)

    ####################
    # KINEMATIC ANIMATION
    ####################

    # Plot path and fit of mechanism
    def plot_path(self,title="Mechanism Path"):
        ax = plt.figure(1)
        ax = plt.gca()
        ax.cla()
        plt.plot(self.path_x,self.path_y)
        plt.plot(self.fit_x,self.fit_y,'r')
        plt.legend(["Path","Linear Fit"])
        plt.title(title)
        plt.grid()
        plt.show()

    # Plots error of mechanism
    def plot_error(self,**kwargs):
        fig = plt.figure(2)
        label = "Iterations"
        x_values = range(len(self.coef_list))
        # Check for alternative label
        if len(kwargs) != 0:
            label = list(kwargs.keys())[0]
            x_values = kwargs[label]
        # Resize figure
        ax = plt.gca()
        ax.cla()
        fig.set_size_inches(10, 12)
        # Plot Convergence of MSE
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
    def draw(self,animation=False,save=False,path="./images/slm.jpg",debug=False):
        # Determines if drawing for animation or not
        if not animation:
            fig = plt.figure(3)
            ax = plt.gca()
            ax.cla()
        else:
            self.ax.cla()
        # Animation Parameters
        link_size = 5
        dot_size = 50
        path_size = 3
        # Draws path
        plt.plot(self.path_y,self.path_x,linewidth = path_size)
        #plt.plot(self.fit_y,self.fit_x,'r')
        n_p = [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(0,3),(0,4)] # Node pairs
        # Draw all links
        for i in range(self.num_links):
            n0 = n_p[i][0]
            n1 = n_p[i][1]
            plt.plot([self.N[n0][0],self.N[n1][0]],[self.N[n0][1],self.N[n1][1]],
                    linewidth=link_size,c=self.colors[i])
        # Draw all nodes
        for N in self.N:
            plt.scatter(N[0],N[1],s=dot_size,c="black",zorder=10)

        # Set plot limits
        # if self.version == 1:
        #     plt.xlim([-.5*self.A[0], 1.2*max(self.path_y)])
        #     plt.ylim([self.path_x[0], self.path_x[-1]])
        # if self.version == 2:
        #     pass
        # if self.version == 3:
        #     plt.xlim([-1.25*(self.A[0]+self.A[1]), 1.2*max(self.path_y)])
        #     plt.ylim([self.path_x[0], self.path_x[-1]])
        # plt.gca().set_aspect("equal")
        #plt.grid()
        # self.ax.set_aspect("equal")

        # Compare link lengths vs theoretical ones
        if debug:
            dist = lambda V1,V2: sum((V1-V2)**2)**0.5
            A0_calc = dist(self.N[0],self.N[1])
            A1_calc = dist(self.N[1],self.N[2])
            B0_calc = dist(self.N[2],self.N[4])
            B1_calc = dist(self.N[2],self.N[3])
            B2_calc = dist(self.N[4],self.N[5])
            B3_calc = dist(self.N[3],self.N[5])
            C1_calc = dist(self.N[0],self.N[3])
            C0_calc = dist(self.N[0],self.N[4])
            print("A0: ", self.A[0], "Calc: ", A0_calc)
            print("A1: ", self.A[1], "Calc: ", A1_calc)
            print("B0: ", self.B[0], "Calc: ", B0_calc)
            print("B1: ", self.B[1], "Calc: ", B1_calc)
            print("B2: ", self.B[2], "Calc: ", B2_calc)
            print("B3: ", self.B[3], "Calc: ", B3_calc)
            print("C0: ", self.C[0], "Calc: ", C0_calc)
            print("C1: ", self.C[1], "Calc: ", C1_calc)
        # Saves figure to specified path
        if save:
            fig.set_size_inches(4, 4)
            fig.tight_layout()
            ax.axis('off')
            plt.savefig(path,dpi=200)
        if not animation:
            plt.show()
        else:
            return plt
    
    # Used to update frames ion animation
    def update_frame(self,theta):
        self.theta = theta
        self.update_state()
        return self.draw(animation=True)

    # Animates mechanism moving across theta range
    def animate(self,theta_range = None):
        if theta_range == None:
            theta_range = [-self.LIMIT_THETA,self.LIMIT_THETA]
        # Plotting Variables
        self.fig = plt.figure(3)
        self.ax = plt.gca()
        self.slm_animation = animation.FuncAnimation(self.fig, self.update_frame, frames=np.append(np.arange(theta_range[0], 
                            theta_range[1], 0.025),np.arange(theta_range[1], theta_range[0], -0.025)), interval=100, repeat=True)
        plt.show()

    ####################
    # STIFFNESS ANIMATION
    ####################

    # Draws mechanism and path in for its current state
    def draw_stiffness(self,d,frame_index,animation=False):
        # Determines if drawing for animation or not
        if not animation:
            fig = plt.figure(3)
        # Animation Parameters
        link_size = 6
        node_size = 8
        dot_size = 4
        path_size = 3
        arrow_size = 5
        stiff_color = "green"
        deflect_color = "C0"
        normal_color = "black"
        alpha_value = 0.5
        pad = 0.1
        # Draws path and deformed path
        gs = gridspec.GridSpec(2, 5)
        ax0 = plt.subplot(gs[0:2,0:3])
        ax0.set_xlabel("Y [m]")
        ax0.set_ylabel("X [m]")
        ax0.plot(self.path_y,self.path_x,color = normal_color,linewidth=path_size)
        ax0.plot(self.k_path_y,self.k_path_x,color=deflect_color,linewidth=path_size,alpha=alpha_value)
        n_p = [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(0,3),(0,4)] # Node pairs
        dN = self.find_deformed_nodes(d,self.N) # Deformation pairs
        # Draw all links and deformed structure
        for i in range(self.num_links):
            n0 = n_p[i][0]
            n1 = n_p[i][1]
            # Underformed
            ax0.plot([self.N[n0][0],self.N[n1][0]],[self.N[n0][1],self.N[n1][1]],linewidth=link_size,c=self.colors[i])
            # # Deferomed
            ax0.plot([dN[n0][0],dN[n1][0]],[dN[n0][1],dN[n1][1]],linewidth=link_size,c=self.colors[i],alpha=alpha_value)
        #Draw all underformed nodes nodes
        for N in self.N:
            ax0.scatter(N[0],N[1],linewidth=node_size,c="black",zorder=10)
        #Draw all nodes and deformed nodes
        for N in dN:
            ax0.scatter(N[0],N[1],linewidth=node_size,c="black",zorder=10,alpha=alpha_value)
        # ax0.set_aspect("equal")    
        if self.version == 1:
            ax0.set_xlim([-.1*self.A[0], self.A[0]+self.A[1]+1.75*max(self.B)])
            ax0.set_ylim([min(self.path_x[0],self.k_path_x[0]), max(self.path_x[-1],self.k_path_x[-1])])
        if self.version == 3:
            ax0.set_xlim([-1.25*(self.A[0]+self.A[1]), max(self.B)])
            ax0.set_ylim([self.path_x[0], self.path_x[-1]])
        ax0.set_aspect("equal")
        ax0.grid()
        ax0.set_title("SLM Path Deflection",fontweight='bold',fontsize=16)
        # Plot the stiffness
        ax1 = plt.subplot(gs[0,3:5])
        ax1.plot(self.k_path_x,self.ky_array,linewidth=path_size,color=stiff_color,alpha=alpha_value)
        ax1.scatter(self.k_path_x[frame_index],self.ky_array[frame_index],color=stiff_color,linewidth=dot_size,zorder=10)
        ax1.set_title("Mechanism Stiffness",fontweight='bold',fontsize=16)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Stiffness [N/m]")
        ax1.set_ylim([0,1.1*max(self.ky_array)])
        ax1.set_xlim([-1.1*max(self.k_path_x),1.1*max(self.k_path_x)])
        ax1.grid()
        # Plot the path
        ax2 = plt.subplot(gs[1,3:5])
        ax2.plot(self.k_path_x,self.k_path_y,color=deflect_color,linewidth=path_size,alpha=alpha_value)
        ax2.set_title("End Point Path",fontweight='bold',fontsize=16)
        ax2.scatter(self.k_path_x[frame_index],self.k_path_y[frame_index],color=deflect_color,linewidth=dot_size,zorder=10)
        height, width = max(self.k_path_y)-min(self.k_path_y), max(self.k_path_x)-min(self.k_path_x)
        ax2.set_ylim([min(self.k_path_y)-pad*height,max(self.k_path_y)+pad*height])
        ax2.set_xlim([min(self.k_path_x)-pad*width,max(self.k_path_x)+pad*width])
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.grid()
        # Improve spacing of sublpots
        self.fig.tight_layout()
        if not animation:
            plt.tight_layout()
            plt.show()
        else:
            return plt

    # Plots stiffness of mechanism over range of path
    def plot_stiffness(self,title="Mechanism Stiffness vs Position",save=False,path="./images/stiffness.jpg"):
        fig = plt.figure(4)
        ax = plt.gca()
        ax.cla()
        plt.plot(self.k_path_x,self.ky_array,linewidth=4)
        if save:
            ax.axis('off')
            plt.grid(False)
            fig.set_size_inches(4, 4)
            fig.tight_layout()
            plt.savefig(path,dpi=200)
        ax.axis('on')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Stiffness [N/m]")
        plt.grid()
        plt.show()


    # Plot path and fit of mechanism
    def plot_path_stiffness(self,title="Stiffness Path"):
        ax = plt.figure(6)
        ax = plt.gca()
        ax.cla()
        plt.plot(self.k_path_x,self.k_path_y)
        plt.title(title)
        plt.grid()
        plt.show()
    
    # Used to update frames ion animation
    def update_frame_k(self,frame_index):
        self.theta = self.theta_array[frame_index]
        d = self.d_array[frame_index] 
        self.update_state()
        return self.draw_stiffness(d,frame_index,animation=True)

    # Animates structural response to loading
    def animate_stiffness(self,theta_range = None,save=False,path = "./videos/stiffness_animation"):
        if theta_range == None:
            theta_range = [-self.LIMIT_THETA,self.LIMIT_THETA]
            # Plotting Variables
            self.fig = plt.figure(3,figsize=(14, 8))
            self.fig = plt.figure(3,figsize=(11.2, 6.4))
            self.fig = plt.figure(3,figsize=(10.5, 6))
            self.fig.set_dpi(80)
            self.slm_animation = animation.FuncAnimation(self.fig, self.update_frame_k, frames=np.append(np.arange(0,len(self.theta_array),1),
                                np.arange(len(self.theta_array)-1,0,-1)), interval=100, repeat=True)
            if save:
                print("SAVING VIDEO")
                self.slm_animation.save(path+".gif", writer='imagemagick', fps=60)
            plt.show()
