import numpy as np
import math
import matplotlib.pyplot as plt

def equations(x, state):
    u, v = state
    u_prime = v
    v_prime = 1 - (2+u**2)*u / (1+u**2)
    return np.array([u_prime, v_prime])

# Function performing one "RK4 step"
def rk4_step(equations, cond, x):
    k1 = dx * equations(x, cond)
    k2 = dx * equations(x + dx/2, cond + k1/2)
    k3 = dx * equations(x + dx/2, cond + k2/2)
    k4 = dx * equations(x + dx, cond + k3)
    return cond + (k1 + 2*k2 + 2*k3 + k4) / 6

# Function performing all the RK4 steps
def solve_ode(equations, initial_cond, x, u_at_x2):
    cond = np.zeros((n_points, len(initial_cond)))  # matrix nX3
    cond[0,:] = initial_cond

    for i in range(1, n_points):
        cond[i,:] = rk4_step(equations, cond[i - 1,:], x[i - 1])
        if x[i] > 1.999999 and x[i] < 2.0001:
            u_at_x2.append(cond[i,0])
            #print(u_at_x2)
    return cond 

# Function implementing linear interpolation
def linear_interpolation(s_vector, u_at_x2, starting_point1, starting_point2):
    u_at_x2 = u_at_x2 - 3.0

    x1, x2 = starting_point1, starting_point2
    u1, u2 = u_at_x2[index1], u_at_x2[index2]

    counter = 0

    # Linear interpolation formula
    x3 = x2 - ((x2 - x1) * u2) / (u2 - u1)
    f3 = np.interp(x3, s_vector, u_at_x2)  # approximate the function value at x3

    while abs(f3) > tolerance:
        counter += 1

        if u1*f3 < 0:
            x2 = x3
            u2 = f3
        elif u2*f3 < 0:
            x1 = x3
            u1 = f3
        elif f3 == 0:
            print(f"The sought positive root is {x3}")
            print(f"Solution found in {counter} steps")
            return x3, counter

        # Update x3 using linear interpolation formula
        x3 = x2 - ((x2 - x1) * u2) / (u2 - u1)
        f3 = np.interp(x3, s_vector, u_at_x2)

    print(f"The sought positive root is s = {x3}")
    print(f"f(s) = {f3}")
    print(f"Number of steps: {counter}\n")

    return x3, counter

def solve_by_shooting(equations, u_0, index1, index2, s_vector):
    u_at_x2 = []  # store values of u at the second condition for each s considered
    c = 0

    for s in s_vector:
        c = c + 1
        state_0 = [u_0, s]  # initial state: (u, v)
        cond = solve_ode(equations, state_0, x, u_at_x2)
        if c%9 == 0:
            print(c)

    starting_point1 = s_vector[index1]  
    starting_point2 = s_vector[index2] 
    u_at_x2 = np.array(u_at_x2)
    print(np.size(u_at_x2), np.size(s_vector))

    root, steps = linear_interpolation(s_vector, u_at_x2, starting_point1, starting_point2)
    return root, steps


x_start = 0.0
x_end = 3.0
n_points = 30000  # so that dx=10^-4
dx = (x_end - x_start) / n_points
x = np.linspace(x_start, x_end, n_points)
print(x)
u_0 = 0.0
tolerance = 1e-5  # tolerance for linear interpolation

# Starting points for interpolation (values of s)
index1 = 0
index2 = 99

s_vector = np.array([s for s in np.arange(-5, 5, 0.1)])

root, steps = solve_by_shooting(equations, u_0, index1, index2, s_vector)

u_at_x2 = [] 
state_0 = [u_0, root]  # initial state: (u, v)
cond = solve_ode(equations, state_0, x, u_at_x2)

x_shooting = np.linspace(x_start, x_end, n_points)
y_shooting = cond[:,0]

fig = plt.figure()
plt.plot(x_shooting, y_shooting)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
