import numpy as np
import math
import matplotlib.pyplot as plt

# Pendulum equations
def pendulum_eq(state):
    theta, y = state
    d_theta = y
    d_y = - g/l1 * math.sin(theta)
    return np.array([d_theta, d_y])

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


g = 9.81  # gravitational acceleration (m/s^2)
l1 = 1.0  # length of the first pendulum (m)

# Initial conditions
theta1_0 =np.pi/4  # initial angle of the first pendulum (rad)
omega1_0 = 0.0  # initial angular velocity of the first pendulum (rad/s)

t_start = 0  # start time (s)
t_end = 15  # end time (s)
n_points = 15000  # number of time points
dt = (t_end - t_start) / n_points  # time step

# Initial state
state_0 = [theta1_0, omega1_0]

t = np.linspace(t_start, t_end, n_points)

# Solve the system of ODEs
state = solve_pendulum(state_0)

# Extract solutions
theta1 = state[:, 0]

# Convert to Cartesian coordinates
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)

fig = plt.figure()
plt.plot(t,x1)
plt.xlabel("Time [t]")
plt.ylabel("x(t)")
plt.grid()
plt.show()

# Phase diagram
fig = plt.figure()
plt.plot(x1, y1)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


# Animate the double pendulum
import matplotlib.animation as animation

# Convert to Cartesian coordinates
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid()

# Initialize pendulum line
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, x1[frame]], [0, y1[frame]])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1)
plt.show()
