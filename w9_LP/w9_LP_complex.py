from scipy import optimize
import numpy as np
c = np.random.rand(1000000)
A_ub = np.random.rand(10,1000000)
b_ub = np.random.rand(10)

res = optimize.linprog(c,A_ub,b_ub) 
print(res)
print("Optimal result is",res.fun)
