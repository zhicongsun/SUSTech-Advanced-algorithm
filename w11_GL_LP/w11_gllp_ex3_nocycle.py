import numpy
from scipy import optimize

c = numpy.array([0,0,0,0,1])

A_ub = numpy.array([[1,0,0,0,-1],[0,1,1,0,-1],[0,0,0,1,-1]])
b_ub = numpy.array([-29,0,-10])
A_eb = numpy.array([[1,1,0,0,0],[0,0,1,1,0]])
b_eb = numpy.array([12,12]) 
all_bounds = (0,None)
res = optimize.linprog(c,A_ub,b_ub,A_eb,b_eb,
    bounds=(all_bounds,all_bounds,all_bounds,all_bounds,all_bounds)) 
print(res)
print("Optimal result is",res.fun)
print("x=[%f,%f,%f,%f,%f] " % (res.x[0],res.x[1],res.x[2],res.x[3],res.x[4]))
