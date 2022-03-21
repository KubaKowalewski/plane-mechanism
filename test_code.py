
# # Import the packages and classes needed in this example:
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Create a numpy array of data:
# x = np.array([6, 16, 26, 36, 46, 56]).reshape((-1, 1))
# print(x)
# y = np.array([4, 23, 10, 12, 22, 35])

# # Create an instance of a linear regression model and fit it to the data with the fit() function:
# model = LinearRegression().fit(x, y) 

# # The following section will get results by interpreting the created instance: 

# # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
# r_sq = model.score(x, y)
# print('coefficient of determination:', r_sq)

# # Print the Intercept:
# print('intercept:', model.intercept_)

# # Print the Slope:
# print('slope:', model.coef_) 

# # Predict a Response and print it:
# y_pred = model.predict(x)
# print('Predicted response:', y_pred, sep='\n')

from src.Slm import slm
import numpy as np
import matplotlib.pyplot as plt

### Stiffness of SLM
SCALE = 6
A = np.array([.1,.1]) * SCALE
B = np.array([.1,.1,.1,.1]) * SCALE
C = np.array([.275,.275]) * SCALE
SLM = slm.mechanism(A,B,C)
# Loading parmaters
E = 200e9 
D = 5
F = [0,-1,0,0,0,0,0,0]
N = SLM.calculate_state(SLM.theta)
thetas = SLM.find_link_angles(N)
stiff_res = SLM.calculate_stiffness(F,thetas,E,D)
k_y = stiff_res[0]
L_max = SLM.MAX_L
print(k_y,L_max)
SLM.draw()

# ### Sweep Parameters
# aRange = [0.1,]
