

# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [3,8]
# y_c = [3,8]
# x_s = [4,5,7]
# y_s = [4,5,7]
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,13))
# plt.yticks(range(0,13))
# # plt.title('r(C) = ')
# plt.grid()
# plt.show()

# # use for example1
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [2,6]
# y_c = [2,6]
# x_s = [1,3,5]
# y_s = [1,3,5]
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,8))
# plt.yticks(range(0,8))
# # plt.title('r(C) = ')
# plt.grid()
# plt.show()

# # example2.1
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [2,1,11]
# y_c = [2,11,1]
# x_s = [1,2,2,3,  1,2,10,11]
# y_s = [2,3,1,2,  10,11,1,2]
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,13))
# plt.yticks(range(0,13))
# plt.grid()
# plt.show()

# # example 2.2
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [1,2,11]
# y_c = [2,11,1]
# x_s = [2,2,2,3,  1,1,10,11]
# y_s = [2,3,1,2,  10,11,1,2]
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,13))
# plt.yticks(range(0,13))
# plt.grid()
# plt.show()

# # use for old example1
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [4,4]
# y_c = [5,3-3**0.5]
# x_s = [4,3,5]
# y_s = [6,3,3]
# # x_s = [4,3,5,4,4]
# # y_s = [5,3,3,6,3-3**0.5]
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,9))
# plt.yticks(range(0,9))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example1.0
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_s = [1,1,3,3,  2, 6,6,8,8,  7]
# y_s = [1,3,1,3,  2, 6,8,6,8,  7]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example1.1 optimal one
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [2,7]
# y_c = [2,7]
# x_s = [1,1,3,3,   6,6,8,8]
# y_s = [1,3,1,3,   6,8,6,8]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example1.2 optimal one
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [2,8]
# y_c = [2,8]
# x_s = [1,1,3,3,   6,6,8,7]
# y_s = [1,3,1,3,   6,8,6,7]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example1.3 optimal one
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [1,8]
# y_c = [3,8]
# x_s = [1,2,3,3,   6,6,8,7]
# y_s = [1,2,1,3,   6,8,6,7]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example2.0 always the optimal one
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_s = [3,4,6,7]
# y_s = [3,4,6,7]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example2.1 always the optimal one
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [3,7]
# y_c = [3,7]
# x_s = [4,6]
# y_s = [4,6]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# # example2.2 always the optimal one
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8))
# x_c = [4,7]
# y_c = [4,7]
# x_s = [3,6]
# y_s = [3,6]
# plt.scatter(x_s,y_s,s=300,c='blue')
# plt.scatter(x_c,y_c,s=300,c='red')
# plt.xticks(range(0,10))
# plt.yticks(range(0,10))
# plt.tick_params(labelsize=30)
# plt.grid()
# plt.show()

# example3.0
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
x_s = [2,2,2,3,1, 4,5,6,  9,8,9]
y_s = [2,1,3,2,2, 5,5,5,  9,9,8]
plt.scatter(x_s,y_s,s=300,c='blue')
plt.xticks(range(0,11))
plt.yticks(range(0,11))
plt.tick_params(labelsize=30)
plt.grid()
plt.show()

# example3.1
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
x_c = [5,9, 2]
y_c = [5,9, 1]
x_s = [2,2,3,1, 4,6,  8,9]
y_s = [2,3,2,2, 5,5,  9,8]
plt.scatter(x_s,y_s,s=300,c='blue')
plt.scatter(x_c,y_c,s=300,c='red')
plt.xticks(range(0,11))
plt.yticks(range(0,11))
plt.tick_params(labelsize=30)
plt.grid()
plt.show()

# example3.2
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
x_c = [5,9, 2]
y_c = [5,9, 2]
x_s = [2,2,3,1, 4,6,  8,9]
y_s = [1,3,2,2, 5,5,  9,8]
plt.scatter(x_s,y_s,s=300,c='blue')
plt.scatter(x_c,y_c,s=300,c='red')
plt.xticks(range(0,11))
plt.yticks(range(0,11))
plt.tick_params(labelsize=30)
plt.grid()
plt.show()