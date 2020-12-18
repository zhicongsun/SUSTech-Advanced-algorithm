import numpy
from scipy import optimize

c = numpy.array([1,2,1,2])

A_ub = numpy.array([[-1,-1,0,0],[0,-1,-1,0],[0,0,-1,-1],[-1,0,0,-1]])
b_ub = numpy.array([-1,-1,-1,-1]) 

res = optimize.linprog(c,A_ub,b_ub) 
print(res)
print("Optimal result is",res.fun)
 