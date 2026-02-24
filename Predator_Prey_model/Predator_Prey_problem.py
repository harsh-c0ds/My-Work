import numpy as np
import math
import matplotlib.pyplot as plt

def Lotka_Volterra(state):
    x, y = state
    x_prime = alpha*x - beta*x*y
    y_prime = delta*x*y - gamma*y
    return np.array([x_prime, y_prime])

# Function performing one "RK4 step"
def rk4_step(equations, cond):
    k1 = dt * equations(cond)
    k2 = dt * equations(cond + k1/2)
    k3 = dt * equations(cond + k2/2)
    k4 = dt * equations(cond + k3)
    return cond + (k1 + 2*k2 + 2*k3 + k4) / 6

# Function performing all the RK4 steps
def solve_Lotka_volterra(initial_state):
    state = np.zeros((n_points, len(initial_state)))  # matrix nX3
    state[0,:] = initial_state

    for i in range(1, n_points):
        state[i,:] = rk4_step(Lotka_Volterra, state[i - 1,:])
    return state

alpha = 0.1
beta = 0.02
gamma = 0.1
delta = 0.01

t_start = 0
t_end = 300
n_points = 300000  # number of time points
dt = (t_end - t_start) / n_points  # time step

# Initial condition (x,y,z)
initial_cond = np.array([40,10])  

cond = solve_Lotka_volterra(initial_cond)
x, y = cond[:, 0], cond[:, 1]

t = np.linspace(t_start, t_end, n_points)

fig = plt.figure()
plt.plot(t, x, color="green", label="Pray")
plt.plot(t, y, color="red", label="Predator")
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Population")
plt.show()

# Parametric plot
fig = plt.figure()
plt.plot(x, y)
plt.grid()
plt.xlabel("Prey population")
plt.ylabel("Predator population")
plt.show()
