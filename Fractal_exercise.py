#!/usr/bin/env python
# coding: utf-8

# Our task is to find the roots of a given function in the complex plane via the Newton-Raphson method.
# The iteration scheme is the same as for functions in real space:
# \begin{equation}
#     z_{n+1}=z_n-\frac{f(z)}{f'(z)},\quad\text{where}\quad z\in\mathbb{C}.
# \end{equation}
# 
# Firstly, we consider the following complex-valued function:
# \begin{equation}
#     f(z)=z^3-1.
# \end{equation}
# 
# Thus, we are asked to solve the equation: $z^3-1=0$.
# 
# The iteration procedure in our case reads:
# \begin{equation}
#     z_{n+1}=z_n-\frac{z^3_n-1}{3z^2_n}.
# \end{equation}
# 
# This function has 3 roots as stated by the fundamental theorem of algebra:
# - 1
# - (0.5, 0.8660254040)
# - (0.5, -0.8660254040)

# In[2]:


import cmath   # to deal with complex numbers
import numpy as np
from matplotlib import pyplot as plt 


# In the next section we define de following functions:
# - $\textbf{f}$ evaluates the complex function.
# - $\textbf{Newton\_Raphson}$ evaluates the recurrence relation.
# - $\textbf{solve\_cnewton}$ performs the Newton_Raphson method for a given initial point and returns the coordinates of the starting point, the root (or the last guess if it does not converge), the function evaluated at the root, the number of steps $n$, and $\log_{10}(n)$.

# In[4]:


def f(z):
    return z**3 - 1


def Newton_Raphson(z):
    z_new = z - (z**3 - 1)/(3*z**2)
    return z_new


def solve_cnewton(duplet):
    counter = 0
    
    x_duplet, y_duplet = duplet
    z = complex(x_duplet, y_duplet)   # make the duplet a point of the complex plane
    #print("z=",z)
    
    for i in range(M):
        z_new = Newton_Raphson(z)
        counter = counter + 1
        
        f_new = f(z_new)
    
        if f_new == 0 or abs(z_new - z) < tolerance:   # we are done
            return x_duplet, y_duplet, z_new, f_new, counter, np.log10(counter)
        elif i == M - 1:
            return x_duplet, y_duplet, z_new, f_new, counter, np.log10(counter)

        z = z_new


# In the following section we construct the grid starting from setting the upper left point and the lower right point. In particular we set (-2,2) and (2,-2) as such points.
# 
# We find the root for each point of the grid and construct the scatter plot with the starting points. We mark the starting points with three different colors (black, green and red) on the basis of the kind of root that we find.
# 
# We set 200 as the maximal number of steps for the iterative procedure.

# In[6]:


tolerance = 1e-9
tolerance2 = 1e-6
M = 200   # maximal number of steps
N = 500   # dimension of the grid (NxN square matrix)

grid_point1 = (-2, 2)   # upper left
grid_point2 = (2, -2)   # lower right
x1, y1 = grid_point1
x2, y2 = grid_point2

x_values = np.linspace(x1, x2, N)   # create the two "axes" of the grid
y_values = np.linspace(y1, y2, N)

x, y = np.meshgrid(x_values, y_values)   # create coordinates of the grid points
#print("x=",x)
#print("y=",y)
grid = np.dstack((x, y))   # construct the grid, each point of it corresponds to a duplet (x, y)
#print(grid)
#print(grid[N-1][N-1])

x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton(grid[2][1])   # test
print(x_duplet, y_duplet, root, f_root, counter, log_counter)

log_n_matrix = np.zeros((N,N))   # initialize the matrix for storing log10(n)

root1_x, root2_x, root3_x = [], [], []
root1_y, root2_y, root3_y = [], [], []

counter_list = []

for i in range(N):
    for j in range(N):
        x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton(grid[i][j])

        log_n_matrix[i][j] = log_counter 

        counter_list.append(counter)
        
        if abs(root.real - 1) < tolerance2:   # root 1
            root1_x.append(x_duplet)
            root1_y.append(y_duplet)
        elif abs(root.imag - 0.8660254040) < tolerance2:   # root (0.5, 0.8660254040)
            root2_x.append(x_duplet)
            root2_y.append(y_duplet)
        else:   # root (0.5, -0.8660254040)
            root3_x.append(x_duplet)
            root3_y.append(y_duplet)
        '''
        print(f"x0={x_duplet}, y0={y_duplet}, root={root}, f(root)={f_root}, counter={counter}, log={log_counter}")
        print("")
        '''
plt.figure()
plt.scatter(root1_x, root1_y, c="k", marker=".", s=0.4)
plt.scatter(root2_x, root2_y, c="g", marker=".", s=0.4)
plt.scatter(root3_x, root3_y, c="r", marker=".", s=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Black: 1; green: (0.5, 0.86602); red: (0.5, -0.86602)")


# We now plot $\log_{10}(n)$ in the ($x$,$y$)-plane.
# 
# From the plot we see that the convergence is reached for every considered starting point. In fact, the maximum number of iterations that the routine hit is 45.

# In[8]:


print(f"Maximum number of iterations: {max(counter_list)}.")

plt.figure()
plt.pcolormesh(x, y, log_n_matrix, shading="auto", cmap="viridis")
plt.colorbar(label="log10(n)")
plt.xlabel("x")
plt.ylabel("y")


# In this section we repeat the entire procedure zooming in around a specific zone in order to highlight the fractal structure.
# In particular we set (0.355,-1.010) and (0.371,-1.045) as the new upper left and lower right points respectively.

# In[10]:


grid_point1 = (0.355, -1.010)   # upper left
grid_point2 = (0.371, -1.045)   # lower right
x1, y1 = grid_point1
x2, y2 = grid_point2

x_values = np.linspace(x1, x2, N)   # create the two "axes" of the grid
y_values = np.linspace(y1, y2, N)

x, y = np.meshgrid(x_values, y_values)   # create coordinates of the grid points

grid = np.dstack((x, y))   # construct the grid, each point of it corresponds to a duplet (x, y)

log_n_matrix = np.zeros((N,N))   # initialize the matrix for storing log10(n)

root1_x, root2_x, root3_x = [], [], []
root1_y, root2_y, root3_y = [], [], []

for i in range(N):
    for j in range(N):
        x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton(grid[i][j])

        log_n_matrix[i][j] = log_counter 
        
        if abs(root.real - 1) < tolerance2:   # root 1
            root1_x.append(x_duplet)
            root1_y.append(y_duplet)
        elif abs(root.imag - 0.8660254040) < tolerance2:   # root (0.5, 0.8660254040)
            root2_x.append(x_duplet)
            root2_y.append(y_duplet)
        else:   # root (0.5, -0.8660254040)
            root3_x.append(x_duplet)
            root3_y.append(y_duplet)
        '''
        print(f"x0={x_duplet}, y0={y_duplet}, root={root}, f(root)={f_root}, counter={counter}, log={log_counter}")
        print("")
        '''
plt.figure()
plt.scatter(root1_x, root1_y, c="k", marker=".", s=0.4)
plt.scatter(root2_x, root2_y, c="g", marker=".", s=0.4)
plt.scatter(root3_x, root3_y, c="r", marker=".", s=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Black: 1; green: (0.5, 0.86602); red: (0.5, -0.86602)")


# In the next sections we repeat the procedure for the following complex-valued function:
# \begin{equation}
# f(z)=35z^9-180z^7+378z^5-420z^3+315z.
# \end{equation}
# 
# This equation has 9 roots:
# - 0
# - (0.93774544, 0.65437520)
# - (0.93774544, -0.65437520)
# - (-0.93774544, 0.65437520)
# - (-0.93774544, -0.65437520)
# - (-1.48569, 0.295006)
# - (-1.48569, -0.295006)
# - (1.48569, 0.295006)
# - (1.48569, -0.295006)
# 
# The code structure is the same as before.
# 
# We set 500 as the maximal number of steps for the iterative procedure.

# In[12]:


def f2(z):
    return 35 * z**9 - 180 * z**7 + 378 * z**5 - 420 * z**3 + 315 * z


def Newton_Raphson2(z):
    z_new = z - (35 * z**9 - 180 * z**7 + 378 * z**5 - 420 * z**3 + 315 * z)/(35 * 9 * z**8 - 180 * 7 * z**6 + 378 * 5 * z**4 - 420 * 3 * z**2 + 315)
    return z_new


def solve_cnewton2(duplet):
    counter = 0
    
    x_duplet, y_duplet = duplet
    z = complex(x_duplet, y_duplet)   # make the duplet a point of the complex plane
    #print("z=",z)
    
    for i in range(M):
        z_new = Newton_Raphson2(z)
        counter = counter + 1
        
        f_new = f2(z_new)
    
        if f_new == 0 or abs(z_new - z) < tolerance:   # we are done
            return x_duplet, y_duplet, z_new, f_new, counter, np.log10(counter)
        elif i == M - 1:
            return x_duplet, y_duplet, z_new, f_new, counter, np.log10(counter)

        z = z_new


# In[13]:


tolerance = 1e-9
tolerance2 = 1e-4
M = 500   # maximal number of steps
N = 500   # dimension of the grid (NxN square matrix)

grid_point1 = (-2, 2)   # upper left
grid_point2 = (2, -2)   # lower right
x1, y1 = grid_point1
x2, y2 = grid_point2

x_values = np.linspace(x1, x2, N)   # create the two "axes" of the grid
y_values = np.linspace(y1, y2, N)

x, y = np.meshgrid(x_values, y_values)   # create coordinates of the grid points
#print("x=",x)
#print("y=",y)
grid = np.dstack((x, y))   # construct the grid, each point of it corresponds to a duplet (x, y)
#print(grid)
#print(grid[N-1][N-1])

x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton2(grid[2][1])   # test
print(x_duplet, y_duplet, root, f_root, counter, log_counter)

log_n_matrix = np.zeros((N,N))   # initialize the matrix for storing log10(n)

root1_x, root2_x, root3_x, root4_x, root5_x, root6_x, root7_x, root8_x, root9_x = [], [], [], [], [], [], [], [], []
root1_y, root2_y, root3_y, root4_y, root5_y, root6_y, root7_y, root8_y, root9_y = [], [], [], [], [], [], [], [], []

counter_list = []

for i in range(N):
    for j in range(N):
        x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton2(grid[i][j])

        log_n_matrix[i][j] = log_counter 

        counter_list.append(counter)
        
        if abs(root.real - 0) < tolerance2:   # root 0
            root1_x.append(x_duplet)
            root1_y.append(y_duplet)
        elif abs(root.real - 0.93774544) < tolerance2 and abs(root.imag - 0.65437520) < tolerance2:   # root (0.93774544, 0.65437520)
            root2_x.append(x_duplet)
            root2_y.append(y_duplet)
        elif abs(root.real - 0.93774544) < tolerance2 and abs(root.imag + 0.65437520) < tolerance2:   # root (0.93774544, -0.65437520)
            root3_x.append(x_duplet)
            root3_y.append(y_duplet)
        elif abs(root.real + 0.93774544) < tolerance2 and abs(root.imag - 0.65437520) < tolerance2:   # root (-0.93774544, 0.65437520)
            root4_x.append(x_duplet)
            root4_y.append(y_duplet)
        elif abs(root.real + 0.93774544) < tolerance2 and abs(root.imag + 0.65437520) < tolerance2:   # root (-0.93774544, -0.65437520)
            root5_x.append(x_duplet)
            root5_y.append(y_duplet)
        elif abs(root.real + 1.48569) < tolerance2 and abs(root.imag - 0.295006) < tolerance2:   # root (-1.48569, 0.295006)
            root6_x.append(x_duplet)
            root6_y.append(y_duplet)
        elif abs(root.real + 1.48569) < tolerance2 and abs(root.imag + 0.295006) < tolerance2:   # root (-1.48569, -0.295006)
            root7_x.append(x_duplet)
            root7_y.append(y_duplet)
        elif abs(root.real - 1.48569) < tolerance2 and abs(root.imag - 0.295006) < tolerance2:   # root (1.48569, 0.295006)
            root8_x.append(x_duplet)
            root8_y.append(y_duplet)
        else:   # root (1.48569, -0.295006)
            root9_x.append(x_duplet)
            root9_y.append(y_duplet)
        '''
        print(f"x0={x_duplet}, y0={y_duplet}, root={root}, f(root)={f_root}, counter={counter}, log={log_counter}")
        print("")
        '''
plt.figure()
plt.scatter(root1_x, root1_y, c="k", marker=".", s=0.4)
plt.scatter(root2_x, root2_y, c="g", marker=".", s=0.4)
plt.scatter(root3_x, root3_y, c="r", marker=".", s=0.4)
plt.scatter(root4_x, root4_y, c="darkorchid", marker=".", s=0.4)
plt.scatter(root5_x, root5_y, c="orange", marker=".", s=0.4)
plt.scatter(root6_x, root6_y, c="yellow", marker=".", s=0.4)
plt.scatter(root7_x, root7_y, c="cyan", marker=".", s=0.4)
plt.scatter(root8_x, root8_y, c="w", marker=".", s=0.4)
plt.scatter(root9_x, root9_y, c="hotpink", marker=".", s=0.4)
plt.xlabel("x")
plt.ylabel("y")


# From the plot below we see that the convergence is reached for every considered starting point. In fact, the maximum number of iterations that is hit by the routine is 248.

# In[15]:


print(f"Maximum number of iterations: {max(counter_list)}.")

plt.figure()
plt.pcolormesh(x, y, log_n_matrix, shading="auto", cmap="viridis")
plt.colorbar(label="log10(n)")
plt.xlabel("x")
plt.ylabel("y")


# We plot the points of a smaller region.

# In[17]:


grid_point1 = (0.35, 1.1)   # upper left
grid_point2 = (0.6, 0.6)   # lower right
x1, y1 = grid_point1
x2, y2 = grid_point2

x_values = np.linspace(x1, x2, N)   # create the two "axes" of the grid
y_values = np.linspace(y1, y2, N)

x, y = np.meshgrid(x_values, y_values)   # create coordinates of the grid points

grid = np.dstack((x, y))   # construct the grid, each point of it corresponds to a duplet (x, y)

log_n_matrix = np.zeros((N,N))   # initialize the matrix for storing log10(n)

root1_x, root2_x, root3_x, root4_x, root5_x, root6_x, root7_x, root8_x, root9_x = [], [], [], [], [], [], [], [], []
root1_y, root2_y, root3_y, root4_y, root5_y, root6_y, root7_y, root8_y, root9_y = [], [], [], [], [], [], [], [], []

for i in range(N):
    for j in range(N):
        x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton2(grid[i][j])

        log_n_matrix[i][j] = log_counter 
        
        if abs(root.real - 0) < tolerance2:   # root 0
            root1_x.append(x_duplet)
            root1_y.append(y_duplet)
        elif abs(root.real - 0.93774544) < tolerance2 and abs(root.imag - 0.65437520) < tolerance2:   # root (0.93774544, 0.65437520)
            root2_x.append(x_duplet)
            root2_y.append(y_duplet)
        elif abs(root.real - 0.93774544) < tolerance2 and abs(root.imag + 0.65437520) < tolerance2:   # root (0.93774544, -0.65437520)
            root3_x.append(x_duplet)
            root3_y.append(y_duplet)
        elif abs(root.real + 0.93774544) < tolerance2 and abs(root.imag - 0.65437520) < tolerance2:   # root (-0.93774544, 0.65437520)
            root4_x.append(x_duplet)
            root4_y.append(y_duplet)
        elif abs(root.real + 0.93774544) < tolerance2 and abs(root.imag + 0.65437520) < tolerance2:   # root (-0.93774544, -0.65437520)
            root5_x.append(x_duplet)
            root5_y.append(y_duplet)
        elif abs(root.real + 1.48569) < tolerance2 and abs(root.imag - 0.295006) < tolerance2:   # root (-1.48569, 0.295006)
            root6_x.append(x_duplet)
            root6_y.append(y_duplet)
        elif abs(root.real + 1.48569) < tolerance2 and abs(root.imag + 0.295006) < tolerance2:   # root (-1.48569, -0.295006)
            root7_x.append(x_duplet)
            root7_y.append(y_duplet)
        elif abs(root.real - 1.48569) < tolerance2 and abs(root.imag - 0.295006) < tolerance2:   # root (1.48569, 0.295006)
            root8_x.append(x_duplet)
            root8_y.append(y_duplet)
        else:   # root (1.48569, -0.295006)
            root9_x.append(x_duplet)
            root9_y.append(y_duplet)
        '''
        print(f"x0={x_duplet}, y0={y_duplet}, root={root}, f(root)={f_root}, counter={counter}, log={log_counter}")
        print("")
        '''
plt.figure()
plt.scatter(root1_x, root1_y, c="k", marker=".", s=0.4)
plt.scatter(root2_x, root2_y, c="g", marker=".", s=0.4)
plt.scatter(root3_x, root3_y, c="r", marker=".", s=0.4)
plt.scatter(root4_x, root4_y, c="darkorchid", marker=".", s=0.4)
plt.scatter(root5_x, root5_y, c="orange", marker=".", s=0.4)
plt.scatter(root6_x, root6_y, c="yellow", marker=".", s=0.4)
plt.scatter(root7_x, root7_y, c="cyan", marker=".", s=0.4)
plt.scatter(root8_x, root8_y, c="w", marker=".", s=0.4)
plt.scatter(root9_x, root9_y, c="hotpink", marker=".", s=0.4)
plt.xlabel("x")
plt.ylabel("y")


# We plot the points of an even smaller region :)

# In[19]:


grid_point1 = (0.5113, 0.741)   # upper left
grid_point2 = (0.51365, 0.73565)   # lower right
x1, y1 = grid_point1
x2, y2 = grid_point2

x_values = np.linspace(x1, x2, N)   # create the two "axes" of the grid
y_values = np.linspace(y1, y2, N)

x, y = np.meshgrid(x_values, y_values)   # create coordinates of the grid points

grid = np.dstack((x, y))   # construct the grid, each point of it corresponds to a duplet (x, y)

log_n_matrix = np.zeros((N,N))   # initialize the matrix for storing log10(n)

root1_x, root2_x, root3_x, root4_x, root5_x, root6_x, root7_x, root8_x, root9_x = [], [], [], [], [], [], [], [], []
root1_y, root2_y, root3_y, root4_y, root5_y, root6_y, root7_y, root8_y, root9_y = [], [], [], [], [], [], [], [], []

for i in range(N):
    for j in range(N):
        x_duplet, y_duplet, root, f_root, counter, log_counter = solve_cnewton2(grid[i][j])

        log_n_matrix[i][j] = log_counter 
        
        if abs(root.real - 0) < tolerance2:   # root 0
            root1_x.append(x_duplet)
            root1_y.append(y_duplet)
        elif abs(root.real - 0.93774544) < tolerance2 and abs(root.imag - 0.65437520) < tolerance2:   # root (0.93774544, 0.65437520)
            root2_x.append(x_duplet)
            root2_y.append(y_duplet)
        elif abs(root.real - 0.93774544) < tolerance2 and abs(root.imag + 0.65437520) < tolerance2:   # root (0.93774544, -0.65437520)
            root3_x.append(x_duplet)
            root3_y.append(y_duplet)
        elif abs(root.real + 0.93774544) < tolerance2 and abs(root.imag - 0.65437520) < tolerance2:   # root (-0.93774544, 0.65437520)
            root4_x.append(x_duplet)
            root4_y.append(y_duplet)
        elif abs(root.real + 0.93774544) < tolerance2 and abs(root.imag + 0.65437520) < tolerance2:   # root (-0.93774544, -0.65437520)
            root5_x.append(x_duplet)
            root5_y.append(y_duplet)
        elif abs(root.real + 1.48569) < tolerance2 and abs(root.imag - 0.295006) < tolerance2:   # root (-1.48569, 0.295006)
            root6_x.append(x_duplet)
            root6_y.append(y_duplet)
        elif abs(root.real + 1.48569) < tolerance2 and abs(root.imag + 0.295006) < tolerance2:   # root (-1.48569, -0.295006)
            root7_x.append(x_duplet)
            root7_y.append(y_duplet)
        elif abs(root.real - 1.48569) < tolerance2 and abs(root.imag - 0.295006) < tolerance2:   # root (1.48569, 0.295006)
            root8_x.append(x_duplet)
            root8_y.append(y_duplet)
        else:   # root (1.48569, -0.295006)
            root9_x.append(x_duplet)
            root9_y.append(y_duplet)
        '''
        print(f"x0={x_duplet}, y0={y_duplet}, root={root}, f(root)={f_root}, counter={counter}, log={log_counter}")
        print("")
        '''
plt.figure()
plt.scatter(root1_x, root1_y, c="k", marker=".", s=0.4)
plt.scatter(root2_x, root2_y, c="g", marker=".", s=0.4)
plt.scatter(root3_x, root3_y, c="r", marker=".", s=0.4)
plt.scatter(root4_x, root4_y, c="darkorchid", marker=".", s=0.4)
plt.scatter(root5_x, root5_y, c="orange", marker=".", s=0.4)
plt.scatter(root6_x, root6_y, c="yellow", marker=".", s=0.4)
plt.scatter(root7_x, root7_y, c="cyan", marker=".", s=0.4)
plt.scatter(root8_x, root8_y, c="w", marker=".", s=0.4)
plt.scatter(root9_x, root9_y, c="hotpink", marker=".", s=0.4)
plt.xlabel("x")
plt.ylabel("y")

plt.show()