# Code for solving the Lorenz equations (3 coupled ODEs) using RK4.
# We study the chaotic behavior of this system by computing the Lyapunov exponent.

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Lorenz equations:
def lorenz_eq(cond):
    x, y, z = cond[0], cond[1], cond[2]
    x_prime = sigma*(y-x)
    y_prime = x*(r-z) - y
    z_prime = x*y - b*z
    return np.array([x_prime, y_prime, z_prime])

# Function performing one "RK4 step"
def rk4_step(equations, cond):
    k1 = dt * equations(cond)
    k2 = dt * equations(cond + k1/2)
    k3 = dt * equations(cond + k2/2)
    k4 = dt * equations(cond + k3)
    return cond + (k1 + 2*k2 + 2*k3 + k4) / 6

# Function performing all the RK4 steps
def solve_Lorenz_eq(initial_cond):
    cond = np.zeros((n_points, len(initial_cond)))  # matrix nX3
    cond[0,:] = initial_cond

    for i in range(1, n_points):
        cond[i,:] = rk4_step(lorenz_eq, cond[i - 1,:])
    return cond 

sigma = 10
r = 28
b = 8/3

t_start = 0
t_end = 60
n_points = 25000  # number of time points
dt = (t_end - t_start) / n_points  # time step

# Initial condition (x,y,z)
initial_cond = np.array([1,1,1])  

cond = solve_Lorenz_eq(initial_cond)
x, y, z = cond[:, 0], cond[:, 1], cond[:, 2]

t = np.linspace(t_start, t_end, n_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
plt.show()


# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting parameters
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Initialize the plot elements
line, = ax.plot([], [], [], 'o-', lw=1)
trail, = ax.plot([], [], [], '-', lw=1, alpha=0.5)
trail_points = 100000  # Number of points in the trail

# Initialization function
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    trail.set_data([], [])
    trail.set_3d_properties([])
    return line, trail

# Update function for animation
def update(frame):
    line.set_data([x[frame]], [y[frame]])
    line.set_3d_properties([z[frame]])

    start = max(0, frame - trail_points)
    trail.set_data(x[start:frame], y[start:frame])
    trail.set_3d_properties(z[start:frame])
    return line, trail

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, interval=0.01, blit=True)  # change interval to speed up or slow down the animation
plt.show()


# CHAOTIC BEHAVIOUR  #####################################################################################################

# Initial condition (x,y,z)
initial_cond = np.array([1,1,1])  
cond = solve_Lorenz_eq(initial_cond)
x, y, z = cond[:, 0], cond[:, 1], cond[:, 2]

# Initial condition (x,y,z)
initial_cond = np.array([1+1e-9,1,1])  # Slight change 
cond = solve_Lorenz_eq(initial_cond)
x2, y2, z2 = cond[:, 0], cond[:, 1], cond[:, 2]

# Distance between the two trajectories over time
d = np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)
log_d = np.log(d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)

ax.plot(x2, y2, z2, lw=0.5)
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
plt.show()

# Plot of the distance between the two trajectories over time
fig = plt.figure()
plt.plot(t,log_d)
plt.xlabel("Time [s]")
plt.ylabel("log(d)")
plt.title("Distance between the two trajectories over time")
plt.grid()
plt.show()

lyapunov_exp_list = []
for i in range(1, n_points):
    # Compute the Lyapunov exponent
    lyapunov_exp = 1/(t[i]-t_start) * np.log(math.sqrt((x[i]-x2[i])**2 + (y[i]-y2[i])**2 + (z[i]-z2[i])**2) / math.sqrt((x[0]-x2[0])**2 + (y[0]-y2[0])**2 + (z[0]-z2[0])**2))
    lyapunov_exp_list.append(lyapunov_exp)

fig = plt.figure()
plt.plot(t[1:], lyapunov_exp_list)
plt.xlabel("Time [s]")
plt.ylabel("Lyapunov exponent")
plt.grid()
plt.show()

lyapunov_exp = 1/(t_end-t_start) * np.log(math.sqrt((x[-1]-x2[-1])**2 + (y[-1]-y2[-1])**2 + (z[-1]-z2[-1])**2) / math.sqrt((x[0]-x2[0])**2 + (y[0]-y2[0])**2 + (z[0]-z2[0])**2))
print(f"Lyapunov exponent = {lyapunov_exp}")


# DIFFERENT STEPSIZES AND INITIAL CONDITIONS ##################################################################################################

t_start = 0
t_end = 60
n_points = 6*10**6  # number of time points
dt = (t_end - t_start) / n_points  # stepsize = 1e-6
t_1 = np.linspace(t_start, t_end, n_points)

# Initial condition (x,y,z)
initial_cond = np.array([1,1,1])  
cond = solve_Lorenz_eq(initial_cond)
x, y, z = cond[:, 0], cond[:, 1], cond[:, 2]

# Initial condition (x,y,z)
initial_cond = np.array([1+5e-15,1,1])  # Slight change 
cond = solve_Lorenz_eq(initial_cond)
x2, y2, z2 = cond[:, 0], cond[:, 1], cond[:, 2]

# Distance between the two trajectories over time
d = np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)
log_d_1 = np.log(d)

# Lower resolution
n_points = 6*10**4  # number of time points
dt = (t_end - t_start) / n_points  # stepsize = 1e-3
t_2 = np.linspace(t_start, t_end, n_points)

# Initial condition (x,y,z)
initial_cond = np.array([1,1,1])  
cond = solve_Lorenz_eq(initial_cond)
x, y, z = cond[:, 0], cond[:, 1], cond[:, 2]

# Initial condition (x,y,z)
initial_cond = np.array([1+5e-15,1,1])  # Slight change 
cond = solve_Lorenz_eq(initial_cond)
x2, y2, z2 = cond[:, 0], cond[:, 1], cond[:, 2]

# Distance between the two trajectories over time
d = np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)
log_d_2 = np.log(d)

# Plot of the distance between the two trajectories over time
fig = plt.figure()
plt.plot(t_1,log_d_1,label="h=1e-6")
plt.plot(t_2,log_d_2,label="h=1e-3")
plt.xlabel("Time [s]")
plt.ylabel("log(d)")
plt.title("Distance between the two trajectories over time")
plt.legend()
plt.grid()
plt.show()
