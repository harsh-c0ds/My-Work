import numpy as np
import math
import matplotlib.pyplot as plt

# Polytropic EoS
def EoS(k, rho, Gamma):
    return k * rho**Gamma

# Lane-Emden equation
def LE_eq(current_state, xi, n):
    theta, x = current_state
    dtheta_dxi = x
    dx_dxi = -theta**n - 2/xi * x
    return np.array([dtheta_dxi, dx_dxi])

# Stellar structure solver
def make_star(n, xi):
    initial_cond = [1, 0]  # boundary conditions: theta(0)=1, x(0)=dtheta/dxi(0)=0
    cond = np.zeros((n_points, len(initial_cond)))  # matrix nXm
    cond[0,:] = initial_cond

    # RK4 steps
    for i in range(1, n_points):
        #print(cond[i-1,0])
        k1 = dxi * LE_eq(cond[i - 1,:], xi[i - 1], n)
        k2 = dxi * LE_eq(cond[i - 1,:] + k1/2, xi[i - 1], n)
        k3 = dxi * LE_eq(cond[i - 1,:] + k2/2, xi[i - 1], n)
        k4 = dxi * LE_eq(cond[i - 1,:] + k3, xi[i - 1], n)
        cond[i,:] = cond[i - 1,:] + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Find xi and M: r=R if P(r)=0 ---> theta=0
        if cond[i,0] < 1e-3:
            R_xi = xi[i]
            break
    return cond, R_xi

# Compute the mass of the star by using Simpson 1/3 rule for integration; result expressed in solar masses
# x: array containing equally spaced points
# y: array to be integrated
# h: spacing between points on x axis
def star_mass(xi, theta, dxi, n):
    h = dxi
    x = xi
    y = theta**n * xi**2
    factor = 4 * np.pi * rho_c * alpha**3
    t1 = 0
    for i in range(len(x)):
        if i == 0 or i == n-1: # necessary conditions for choosing the coefficients according to the Simpson's rule
            t1 += y[i]
        elif i % 2 == 0:
            t1 += 2*y[i]
        else:
            t1 += 4*y[i]
    return factor * (h/3) * (t1)  # Simpson 1/3 rule
       
# Adiabatic index 
n = 1
Gamma = 1 + 1 / n

# Conversion factors
rho_conversion_factor = 6.178 * 10**17  # [g/cm^3]
r_conversion_factor = 1.477  # [km]; for conversion from IS to natural units
time_conversion_factor = 4.926 * 10**(-6)  # [s], G*solar mass / c^3

G = 1  # Gravitational Constant in cgs
k = 100  # natural units
k_IS = 4.121 * 10**(12)  # [m^5 s^-2]
rho_c_cgs = 5e14  # [g/cm^3]
rho_c = rho_c_cgs / rho_conversion_factor
alpha = np.sqrt(k*(n + 1)*rho_c**((1-n)/n)/(4*np.pi*G))
#print("Alpha=",alpha)

r_start_IS = 10e-4  # [km]
r_end_IS = 100  # [km]
r_start = r_start_IS / r_conversion_factor  # [c=G=solar mass=1 units]
r_end = r_end_IS / r_conversion_factor
n_points = 5*10**5  # number of points of the grid
dr = (r_end - r_start) / n_points
r = np.linspace(r_start, r_end, n_points)  # spatial grid [c=G=solar mass=1 units]
r_IS = r * r_conversion_factor  # [km]
dr_IS = (r_end_IS - r_start_IS) / n_points  # [km]

xi_start = r_start / alpha
xi_end = r_end / alpha
xi = np.linspace(xi_start, xi_end, n_points)  # spatial grid
dxi = (xi_end - xi_start) / n_points

# Stellar model
cond, R_xi = make_star(n, xi)
theta = cond[:,0]
x = cond[:,1]

# Cut the vanishing part of theta and x
nonzero_indices = np.nonzero(theta)[0]
last_nonzero = nonzero_indices[-1]
theta = theta[:last_nonzero + 1]
x = x[:last_nonzero + 1]
r = r[:last_nonzero + 1]
r_IS = r_IS[:last_nonzero + 1]

rho = rho_c * theta**n  # density (vector) [c=G=solar mass=1 units]
rho_cgs = rho * rho_conversion_factor  # [g/cm^3]
rho_IS = rho_cgs * 10**3  # [kg/m^3]
#P = EoS(k_IS, rho_IS, Gamma)  # pressure (vector) [IS units]
P = EoS(k, rho, Gamma)  # pressure (vector) [c=G=solar mass=1 units]

R = R_xi * alpha  # radius of the star [c=G=solar mass=1 units]
R_IS = R * r_conversion_factor  # [km]
M = star_mass(xi, cond[:,0], dxi, n)  # mass of the star [solar masses]
#M = M / r_conversion_factor  # [solar mass]
print("Radius of the star [km] =",R_IS)
print("Mass of the star [solar mass] =",M)

# Density profile
fig = plt.figure()
plt.plot(r_IS, rho_cgs)
plt.xlabel("r [km]")
plt.ylabel(r"$\rho$ [g/cm^3]")
plt.title(fr"$\kappa$={k}, n={n}")
plt.grid()
plt.show()

# Pressure profile
fig = plt.figure()
plt.plot(r_IS, P)
plt.xlabel("r [km]")
plt.ylabel("P")
plt.title(fr"$\kappa$={k}, n={n}")
plt.grid()
plt.show()


# STELLAR PULSATIONS #########################################################################################
# In this part we solve the master equation.

# System of ODEs arising from the master equation
def master_eq(current_state, r, P, dP_dr, rho, Gamma1, omega):
    xir, y = current_state
    A = 2/r + 1/(Gamma1*P)*dP_dr 
    B = (omega**2 * rho)/(Gamma1*P) - 4/r**2
    dxir_dr = y
    dy_dr = -A*y - B*xir
    return np.array([dxir_dr, dy_dr])

# Function performing one "RK4 step"
def rk4_step(equations, cond, r, dr, P, dP_dr, rho, Gamma1, omega):
    k1 = dr * equations(cond, r, P, dP_dr, rho, Gamma1, omega)
    k2 = dr * equations(cond + k1/2, r, P, dP_dr, rho, Gamma1, omega)
    k3 = dr * equations(cond + k2/2, r, P, dP_dr, rho, Gamma1, omega)
    k4 = dr * equations(cond + k3, r, P, dP_dr, rho, Gamma1, omega)
    return cond + (k1 + 2*k2 + 2*k3 + k4) / 6

# Function performing all the RK4 steps
def solve_master(initial_state, r, dr, P, dP_dr, rho, Gamma1, omega):
    #state = np.zeros((len(r), len(initial_state)))  # matrix nXm
    #state[0,:] = initial_state
    cond_at_surf_list = []
    for j in range(len(r)-1):
        state = np.zeros((len(r), len(initial_state)))  # matrix nXm
        state[0,:] = initial_state
        current_omega = omega[j]
        for i in range(1, len(r)):
            state[i,:] = rk4_step(master_eq, state[i-1,:], r[i-1], dr, P[i-1], dP_dr[i-1], rho[i-1], Gamma1[i-1], current_omega)
        #print(state)
        cond_at_surf = state[-1,1] + drho_dr[-1]/rho[-1] * state[-1,0]
        print(cond_at_surf)
        cond_at_surf_list.append(cond_at_surf)
        if j != 0:
            if cond_at_surf_list[j]*cond_at_surf_list[j-1] < 0:
                return state, current_omega
    print("The condition at surface is not met.")
    return state, current_omega

# Compute Gamma1: Gamma1 = dlog(P)/dlog(rho)
def compute_Gamma1(P, rho):
    N = len(P)
    Gamma1 = np.zeros(N)
    
    # For the first point
    dlnP_first = math.log(P[1]) - math.log(P[0])
    dlnrho_first = math.log(rho[1]) - math.log(rho[0])
    Gamma1[0] = dlnP_first / dlnrho_first
    
    # Central difference for interior points:
    for i in range(1, N-1):
        dlnP = math.log(P[i+1]) - math.log(P[i-1])
        dlnrho = math.log(rho[i+1]) - math.log(rho[i-1])
        Gamma1[i] = dlnP / dlnrho
    
    # For the last point
    dlnP_end = math.log(P[N-1]) - math.log(P[N-2])
    dlnrho_end = math.log(rho[N-1]) - math.log(rho[N-2])
    Gamma1[N-1] = dlnP_end / dlnrho_end
    return Gamma1

#G_IS = 6.674e-11  # [m^3 kg^-1 s]
#rho_c_IS = rho_c_cgs * 10**3  # [kg/m^3]
#alpha_IS = np.sqrt(k_IS*(n + 1)*rho_c_IS**((1-n)/n)/(4*np.pi*G_IS))

# Compute derivative of density and derivative of pressure
#drho_dr = rho_c_IS/alpha_IS * n * theta**(n-1) * x
#dP_dr = k_IS/alpha_IS * rho_c_IS**(Gamma) * Gamma * n * theta**n * x
drho_dr = rho_c/alpha * n * theta**(n-1) * x
dP_dr = k/alpha * rho_c**(Gamma) * Gamma * n * theta**n * x

# Plot of the two derivatives
fig = plt.figure()
plt.plot(r_IS, drho_dr)
plt.xlabel("r [km]")
plt.ylabel(r"$\frac{d\rho}{dr}$")
plt.grid()
plt.show()

fig = plt.figure()
plt.plot(r_IS, dP_dr)
plt.xlabel("r [km]")
plt.ylabel(r"$\frac{dP}{dr}$")
plt.grid()
plt.show()

# Gamma1 = dlog(P)/dlog(rho)
Gamma1 = compute_Gamma1(P, rho)

initial_state = np.array([0,1e-4])  # [initial xi_r, initial dxir_dr]
initial_omega = 1
final_omega = 3
step = 1e-4
omega = np.array([item for item in np.arange(initial_omega, final_omega, step)])

#r_IS = r_IS * 10**3  # [m]
#dr_IS = dr_IS * 10**3  # [m]

#state, true_omega = solve_master(initial_state, r_IS, dr_IS, P, dP_dr, rho_IS, Gamma1, omega)
state, true_omega = solve_master(initial_state, r, dr, P, dP_dr, rho, Gamma1, omega)
f_IS = true_omega / (2*np.pi*time_conversion_factor) / 10**3  # kHz
print("omega [natural units]=",true_omega)
print(f"f [kHz]={f_IS:.3e}")


# M-R DIAGRAM ########################################################################################
# n=1: we expect to find a vertical line
# n<1: M increases with R
# 1<n<3: M decreases with R
rho_c_list_cgs = np.array([item for item in np.arange(5e14, 3e15, 5e13)])  # [g/cm^3]
rho_c_list = rho_c_list_cgs / rho_conversion_factor  # density in natural units [c=G=solar mass=1 units]
n_list = [0.8, 1, 1.5]  # polytropic index
for n in n_list:
    M_list = []
    R_list = []
    for rho_c in rho_c_list:
        alpha = np.sqrt(k*(n + 1)*rho_c**((1-n)/n)/(4*np.pi*G))
        xi_start = r_start / alpha
        xi_end = r_end / alpha
        xi = np.linspace(xi_start, xi_end, n_points)  # spatial grid
        dxi = (xi_end - xi_start) / n_points

        cond, R_xi = make_star(n, xi)
        R = R_xi * alpha
        R_IS = R * r_conversion_factor  # km
        M = star_mass(xi, cond[:,0], dxi, n)
        R_list.append(R_IS)
        M_list.append(M)
    fig = plt.figure()
    plt.plot(R_list, M_list)
    plt.title(fr"$\kappa$={k}, n={n}")
    plt.xlabel("R [km]")
    plt.ylabel("M [solar mass]")
    plt.grid()
    plt.show()
