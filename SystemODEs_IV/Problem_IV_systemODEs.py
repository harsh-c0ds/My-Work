import numpy as np
import math
import matplotlib.pyplot as plt

def ode(x, state):
    y, z = state
    y_prime = math.sin(y) + math.cos(z*x)
    z_prime =math.exp(-y*x) + math.sin(z*x)/x
    return np.array([y_prime, z_prime])

# Function performing a single "RK4 step"
def rk4_step(func, state, x):
    k1 = dx * func(x, state)
    k2 = dx * func(x + dx/2, state + k1/2)
    k3 = dx * func(x + dx/2, state + k2/2)
    k4 = dx * func(x + dx, state + k3)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Function performing all the RK4 steps
def solve_ode(x, state_0):
    state = np.zeros((n_points, len(state_0)))

    state[0,:] = state_0
    for i in range(1, n_points):
        state[i,:] = rk4_step(ode, state[i - 1,:], x[i - 1])
    return state

x_start = -1  # start time (s)
x_end = 4  # end time (s)
n_points = 100  # number of time points
dx = (x_end - x_start) / n_points  # time step

# Initial condition: (y_0, z_0)
state_0 = [2.37, -3.48]

x = np.linspace(x_start, x_end, n_points)

# Solve the system of ODEs
state = solve_ode(x, state_0)

y = state[:,0]
z = state[:,1]

fig = plt.figure()
plt.plot(x, y, color="blue", label="y(x)")
plt.plot(x, z, color="red", label="z(x)")
plt.xlabel("x(x)")
plt.ylabel("y(x) and z(x)")
plt.legend()
plt.grid()
plt.show()

# Parametric plot
fig = plt.figure()
plt.plot(y, z)
plt.xlabel("y(x)")
plt.ylabel("z(x)")
plt.grid()
plt.show()
