import numpy as np
import random
import matplotlib.pyplot as plt

## Find linear fit and calculate least-square error
num_points = 5
A = np.eye(5) # Coefficient matrix
B = list(range(num_points))
max_noise = 0.1
noise = lambda x: x + random.uniform(-max_noise,max_noise)
B = list(map(noise,B))
X = np.linalg.lstsq(A,B) # Calculate best fit line
print(B)
print(X)
print(type(X))
for elmn in X:
    print(elmn)



# np.ones(num_points)+random
# print(B)
# B = [1,1.1,1.3,2,4]
# # print(np.lookfor('linear fit'))
# # help("numpy.linalg.lstsq")

# line_fit = np.linalg.lstsq(x,y) 
# print(line_fit)
