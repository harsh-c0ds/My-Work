import numpy as np
import math
import matplotlib.pyplot as plt


# SHOOTING METHOD ######################################################################################################à

def equations(x, state):
    u, v = state
    u_prime = v
    v_prime = 2*u
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
    cond = np.zeros((n_points, len(initial_cond)))  # matrix nX2
    cond[0,:] = initial_cond

    for i in range(1, n_points):
        cond[i,:] = rk4_step(equations, cond[i - 1,:], x[i - 1])
        if x[i] > 0.999999 and x[i] < 1.0001:
            u_at_x2.append(cond[i,0])
            #print(u_at_x2)
    return cond 

# Function implementing linear interpolation
def linear_interpolation(s_vector, u_at_x2, starting_point1, starting_point2):
    u_at_x2 = u_at_x2 - 0.9

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
x_end = 2.0
n_points = 20000  # so that dx=10^-4
dx = (x_end - x_start) / n_points
x = np.linspace(x_start, x_end, n_points)
print(x)
u_0 = 1.2
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


# MATRIX METHOD ###########################################################################################################

# Function performing Gaussian elimination with pivoting
def gaussian_elimination(A, b, tolerance):
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)
    
    for i in range(n - 1):
        if abs(A[i, i]) < tolerance:
            temp = A[i+1:, i]
            index_max = np.argmax(np.abs(temp)) + i + 1  # pivoting
            
            if A[index_max, i] == 0:
                if i == n - 1:
                    break
                else:
                    continue
            
            A[[i, index_max]] = A[[index_max, i]]
            b[i], b[index_max] = b[index_max], b[i]
        
        # Gaussian elimination formula
        for k in range(i + 1, n):
            t1 = A[k, i] / A[i, i]
            A[k, :] = t1 * A[i, :] - A[k, :]
            b[k] = t1 * b[i] - b[k]
    
    x = np.zeros(n)
    x[-1] = b[-1] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        sigma = np.sum(A[i, i+1:] * x[i+1:])
        x[i] = (b[i] - sigma) / A[i, i]
    
    return x

def matrix_method(a, b, y_a, y_b, k2, n, tolerance):
    h = (b - a) / (n + 1)  # step size
    x = np.linspace(a, b, n + 2)  # points where the function y(x) is approximated
    
    A = np.zeros((n, n))

    # System of lynear equations written in matrix form
    for i in range(n):
        A[i, i] = 2 + k2 * h**2
        if i > 0:
            A[i, i - 1] = -1
        if i < n - 1:
            A[i, i + 1] = -1
    
    B = np.zeros(n)
    B[0] = y_a
    B[-1] = y_b
    
    y = gaussian_elimination(A, B, tolerance)
    
    y = np.concatenate(([y_a], y, [y_b]))  # numerical solution within the domain considered
    
    return x, y


tolerance = 10e-6
n = 20  # dimension
a, b = 0, 1  # domain [0,1]
y_a, y_b = 1.2, 0.9  # boundary conditions
k2 = 2  # k^2

x, y = matrix_method(a, b, y_a, y_b, k2, n, tolerance)

fig = plt.figure()
plt.plot(x, y, marker="o", label="Matrix method")
plt.plot(x_shooting, y_shooting, label="Shooting method")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Shooting method vs matrix method")
plt.grid()
plt.legend()
plt.show()