# Code for solving the equation of motion of double pendulum using RK4 method.
# We study the chaotic behavior of this system by computing the Lyapunov exponent.

import numpy as np
import math
import matplotlib.pyplot as plt

# Equations of motion of double pendulum
def double_pendulum_derivatives(y, m1, m2):
    theta1, omega1, theta2, omega2 = y
    delta_theta = theta1 - theta2
    #mu = 1 + m1/m2
    F = (m1 + m2) - m2*np.cos(delta_theta)**2
    theta1_ddot = (1 / (l1*F)) * (g*m2*np.sin(theta2)*np.cos(delta_theta) - g*(m1 + m2)*np.sin(theta1) - (omega2**2 * l2*m2 + omega1**2 * l1*m2*np.cos(delta_theta))*np.sin(delta_theta))
    theta2_ddot = (1 / (l2*F)) * (g*(m1 + m2)*np.sin(theta1)*np.cos(delta_theta) - g*(m1 + m2)*np.sin(theta2) + (omega1**2 * l1*(m1 + m2) + omega2**2 * l2*m2*np.cos(delta_theta))*np.sin(delta_theta))
    #F = mu - np.cos(delta_theta)**2
    #theta1_ddot = 1/(l1*F) * (g*(np.sin(theta2)*np.cos(delta_theta) - mu*np.sin(theta1)) - (omega2**2 * l2 + omega1**2 * l1*np.cos(delta_theta))*np.sin(delta_theta))
    #theta2_ddot = 1/(l2*F) * (g*mu*(np.sin(theta1)*np.cos(delta_theta) - np.sin(theta2)) - (mu*omega1**2 * l1 + omega2**2 * l2*np.cos(delta_theta))*np.sin(delta_theta))
    return np.array([omega1, theta1_ddot, omega2, theta2_ddot])

# Function performing a single "RK4 step"
def rk4_step(func, y, m1, m2):
    k1 = dt * func(y, m1, m2)
    k2 = dt * func(y + k1 / 2, m1, m2)
    k3 = dt * func(y + k2 / 2, m1, m2)
    k4 = dt * func(y + k3, m1, m2)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Function performing all the RK4 steps
def solve_double_pendulum(y0, m1, m2):
    y = np.zeros((n_points, len(y0)))
    y[0,:] = y0
    for i in range(1, n_points):
        y[i,:] = rk4_step(double_pendulum_derivatives, y[i - 1,:], m1, m2)
    return y

g = 9.81  # gravitational acceleration (m/s^2)
l1 = 1.0  # length of the first pendulum (m)
l2 = 1.0  # length of the second pendulum (m)
m1 = 1.0  # mass of the first pendulum (kg)
m2 = 1.0  # mass of the second pendulum (kg)

# Initial conditions
theta1_0 = np.pi/2  # initial angle of the first pendulum (rad)
theta2_0 = np.pi/2  # initial angle of the second pendulum (rad)
omega1_0 = 0.0  # initial angular velocity of the first pendulum (rad/s)
omega2_0 = 0.0  # initial angular velocity of the second pendulum (rad/s)

t_start = 0  # start time (s)
t_end = 70  # end time (s)
n_points = 25000  # number of time points
dt = (t_end - t_start) / n_points  # time step

# Initial state
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

t = np.linspace(t_start, t_end, n_points)

# Solve the system of ODEs
y = solve_double_pendulum(y0, m1, m2)

# Extract solutions
theta1 = y[:, 0]
theta2 = y[:, 2]

# Convert to Cartesian coordinates
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# Plot the angles over time
plt.figure(figsize=(10, 6))
plt.plot(t, theta1, label=r"$\theta_1(t)$", color="blue")
plt.plot(t, theta2, label=r"$\theta_2(t)$", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Double Pendulum Angles Over Time")
plt.legend()
plt.grid()
plt.show()

# Animate the double pendulum
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)
trail, = ax.plot([], [], '-', lw=1, alpha=0.5)
trail_points = 5000  # number of points in the trail

def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line, trail

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    trail.set_data(x2[max(0, frame - trail_points):frame], y2[max(0, frame - trail_points):frame])
    return line, trail

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=0.0001)
plt.show()


# CHAOTIC BEHAVIOUR  ###########################################################################################################

# Initial state
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

y = solve_double_pendulum(y0, m1, m2)

theta1 = y[:, 0]
theta2 = y[:, 2]

# Convert to Cartesian coordinates
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# Initial state
theta2_0 = theta2_0 + 1e-5
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

y = solve_double_pendulum(y0, m1, m2)

theta1 = y[:, 0]
theta2 = y[:, 2]

# Convert to Cartesian coordinates 
x1_2 = l1 * np.sin(theta1)
y1_2 = -l1 * np.cos(theta1)
x2_2 = x1 + l2 * np.sin(theta2)
y2_2 = y1 - l2 * np.cos(theta2)

fig = plt.figure()
plt.plot(x2, y2)
plt.plot(x2_2,y2_2)
plt.grid()
plt.show()

lyapunov_exp_list = []
for i in range(1, n_points):
    # Compute the Lyapunov exponent
    lyapunov_exp = 1/(t[i]-t_start) * np.log(math.sqrt((x2[i] - x2_2[i])**2 + (y2[i] - y2_2[i])**2) / math.sqrt((x2[0] - x2_2[0])**2 + (y2[0] - y2_2[0])**2))
    lyapunov_exp_list.append(lyapunov_exp)

fig = plt.figure()
plt.plot(t[1:], lyapunov_exp_list)
plt.xlabel("Time [s]")
plt.ylabel("Lyapunov exponent")
plt.grid()
plt.show()


# CHANGE RATION m1/m2  ###########################################################################################################

m1 = np.array([mass for mass in np.arange(1, 20, 0.3)])
mass_ratio = m1 / m2

lyapunov = []  # store the Lyapunov exponents

for mass in m1:
    # Initial state
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    y = solve_double_pendulum(y0, mass, m2)

    theta1 = y[:, 0]
    theta2 = y[:, 2]

    # Convert to Cartesian coordinates
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)


    # Initial state
    theta2_0 = theta2_0 + 1e-5
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

    y = solve_double_pendulum(y0, mass, m2)

    theta1 = y[:, 0]
    theta2 = y[:, 2]

    # Convert to Cartesian coordinates 
    x1_2 = l1 * np.sin(theta1)
    y1_2 = -l1 * np.cos(theta1)
    x2_2 = x1 + l2 * np.sin(theta2)
    y2_2 = y1 - l2 * np.cos(theta2)


    # Compute the Lyapunov exponent
    lyapunov_exp = 1/(t_end-t_start) * np.log(math.sqrt((x2[-1] - x2_2[-1])**2 + (y2[-1] - y2_2[-1])**2) / math.sqrt((x2[0] - x2_2[0])**2 + (y2[0] - y2_2[0])**2))
    lyapunov.append(lyapunov_exp)
    #print("Lyapunov exponent = ", lyapunov_exp)

fig = plt.figure()
plt.plot(mass_ratio, lyapunov)
plt.grid()
plt.xlabel("m1/m2")
plt.ylabel("Lyapunov exponent")
plt.show()


# CHANGE STEP SIZE dt  #########################################################################################################

t_start = 0  # start time (s)
t_end = 40  # end time (s)
m1 = 1.0
m2 = 1.0

# Initial conditions
theta1_0 = np.pi/2  # initial angle of the first pendulum (rad)
theta2_0 = np.pi/2  # initial angle of the second pendulum (rad)
omega1_0 = 0.0  # initial angular velocity of the first pendulum (rad/s)
omega2_0 = 0.0  # initial angular velocity of the second pendulum (rad/s)

# Initial state
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

n_points_list = [item for item in range(400, 44400, 2200)]  # number of time points

for n_points in n_points_list:
    dt = (t_end - t_start) / n_points  # time step

    t = np.linspace(t_start, t_end, n_points)

    # Solve the system of ODEs
    y = solve_double_pendulum(y0, m1, m2)

    # Extract solutions
    theta1 = y[:, 0]
    theta2 = y[:, 2]

    # Plot the angles over time
    plt.figure(figsize=(10, 6))
    plt.plot(t, theta1, label=r"$\theta_1(t)$")
    plt.plot(t, theta2, label=r"$\theta_2(t)$")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title(f"dt={dt} (n_points={n_points})")
    plt.legend()
    plt.grid()
    plt.show()
