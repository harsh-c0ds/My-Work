import numpy as np
import math
import matplotlib.pyplot as plt

# Damped oscillator equations
def pendulum_eq(state):
    x, y = state
    d_x = y
    d_y = - 1/m * (k*x + b*y)
    return np.array([d_x, d_y])

# Function performing a single "RK4 step"
def rk4_step(func, state):
    k1 = dt * func(state)
    k2 = dt * func(state + k1 / 2)
    k3 = dt * func(state + k2 / 2)
    k4 = dt * func(state + k3)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Function performing all the RK4 steps
def solve_pendulum(state_0):
    state = np.zeros((n_points, len(state_0)))

    state[0,:] = state_0
    for i in range(1, n_points):
        state[i,:] = rk4_step(pendulum_eq, state[i - 1,:])
    return state

b = 1  # damping factor
k = 300  # spring's constant
m = 1  # (m)

# Initial conditions
x_0 = 1.0  # initial position of the mass (m)
y_0 = 0.0  # initial velocity of the mass (m/s)

t_start = 0  # start time (s)
t_end = 15  # end time (s)
n_points = 15000  # number of time points
dt = (t_end - t_start) / n_points  # time step

# Initial state
state_0 = [x_0, y_0]

t = np.linspace(t_start, t_end, n_points)

# Solve the system of ODEs
state = solve_pendulum(state_0)

x = state[:, 0]
y = np.zeros(len(x))

# Animate the damped oscillator
import matplotlib.animation as animation

# Set up figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid()

# Initialize oscillator line
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, x[frame]], [0, y[frame]])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1)
plt.show()

# Plot x(t) vs time
fig = plt.figure()
plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.grid()
plt.show()
