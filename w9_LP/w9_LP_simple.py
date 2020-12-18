import numpy
from scipy import optimize

c = numpy.array([2,1])

A_ub = numpy.array([[1,1],[-1,-1],[-1,1],[1,-1]])
b_ub = numpy.array([1,1,1,1]) 

res = optimize.linprog(-c,A_ub,b_ub) 
print(res)
print("Optimal result is",-res.fun)


# #A_eq = numpy.array() 
# #B_eq = numpy.array() 
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(-1,2)
# y1 = -x + 1
# y2 = -x - 1
# y3 = x + 1
# y4 = x - 1 
# fig = plt.figure(figsize=(5,5))

# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)
# plt.plot(x,y4)
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.show()

# x = (-None,None) 
# y = (-None,None) 