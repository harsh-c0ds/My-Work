import numpy as np
import math
import matplotlib.pyplot as plt

# Polytropic EoS
def EoS(k, rho_0, Gamma):
    return k * rho_0**Gamma

# TOV equations for P and m
def TOV_eq(current_state, r, rho):
    P, m = current_state
    dp_dr = -(rho + P) * (m + 4 * np.pi * r**3 * P) / (r*(r - 2*m))
    dm_dr = 4 * np.pi * r**2 * rho
    return np.array([dp_dr, dm_dr])

# Stellar structure solver
def make_star(rho_0_c, Gamma, k):
    P_c = EoS(k, rho_0_c, Gamma)
    initial_cond = [P_c, 0]  # boundary conditions: P_c and m(r=0)=0
    cond = np.zeros((n_points, len(initial_cond)))  # matrix nXm
    cond[0,:] = initial_cond

    r = np.linspace(r_start, r_end, n_points)  # spatial grid

    # RK4 steps
    for i in range(1, n_points):
        current_P = cond[i - 1,0]
        if i == 1:
            rho_0 = rho_0_c
        else:
            rho_0 = (current_P / k)**(1 / Gamma)  # compute rho_0 from EoS
        #epsilon = current_P / (rho_0*(Gamma-1))  # specific internal energy
        rho = rho_0 + current_P / (Gamma-1)
        k1 = dr * TOV_eq(cond[i - 1,:], r[i - 1], rho)
        k2 = dr * TOV_eq(cond[i - 1,:] + k1/2, r[i - 1], rho)
        k3 = dr * TOV_eq(cond[i - 1,:] + k2/2, r[i - 1], rho)
        k4 = dr * TOV_eq(cond[i - 1,:] + k3, r[i - 1], rho)
        cond[i,:] = cond[i - 1,:] + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Find R and M: r=R if P(r)=0; M=m(R)
        if cond[i,0] < 1e-7:
            R = r[i]
            M = cond[i,1]
            break
    return M, R
       
# Conversion factors
rho_conversion_factor = 6.18 * 10**17  # [g/cm^3]
r_conversion_factor = 1.476  # km ---> G*solar mass / c^2

r_start_IS = 1e-5  # [km]
r_end_IS = 16  # [km]
r_start = r_start_IS / r_conversion_factor  # c=G=solar mass=1 units
r_end = r_end_IS / r_conversion_factor  # c=G=solar mass=1 units
n_points = 5*10**5  # number of points of the grid
dr = (r_end - r_start) / n_points  # grid step
rho_0_c_SI = 5*10**14  # [g/cm^3]
rho_0_c = rho_0_c_SI / rho_conversion_factor  # 5*10^(14) g/cm^3 (c=G=solar mass=1 units)

# EoS parameters
Gamma = 5/2
k = 3000

M, R = make_star(rho_0_c, Gamma, k)
print(f"Mass={M}, Radius={R} in c=G=solar mass=1 units")


# M-R DIAGRAM ##################################################################################################################

M_list = []
R_list = []
N = 40
rho_0_c_list = np.linspace(7*10**13, 4*10**15, N)  # [g/cm^3]
for rho_0_c_SI in rho_0_c_list:
    rho_0_c = rho_0_c_SI / rho_conversion_factor  
    M, R = make_star(rho_0_c, Gamma, k)
    M_list.append(M)
    R_list.append(R)

M_max = max(M_list)
idx_M_max = M_list.index(M_max)
R_M_max = R_list[idx_M_max]
print(f"Maximum mass={M_max}")  # 1.92 solar masses
print(f"Related radius={R_M_max}")  # 10.13 km

M_list = np.array(M_list)
R_list = np.array(R_list)

# Conversion from natural units to km
R_list_converted = R_list * r_conversion_factor

fig = plt.figure()
plt.plot(R_list_converted, M_list)
plt.xlabel("R [km]")
plt.ylabel("M")
plt.title(rf"M-R diagram, $\Gamma$={Gamma}")
plt.grid()
plt.show()

fig = plt.figure()
plt.plot(rho_0_c_list, M_list)
plt.xlabel(r"$\rho_c$")
plt.ylabel("M")
plt.title(r"M=M($\rho_c$) (c=G=solar mass=1)")
plt.grid()
plt.show()

# DIFFERENT POLYTROPIC EOS
# EoS parameters
Gamma = 2  # n=1
k = 100

M_list = []
R_list = []
N = 40
rho_0_c_list = np.linspace(8*10**13, 4*10**15, N)  # [g/cm^3]
for rho_0_c_SI in rho_0_c_list:
    rho_0_c = rho_0_c_SI / rho_conversion_factor  
    M, R = make_star(rho_0_c, Gamma, k)
    M_list.append(M)
    R_list.append(R)

M_max = max(M_list)
idx_M_max = M_list.index(M_max)
R_M_max = R_list[idx_M_max]
print(f"Maximum mass={M_max}")  
print(f"Related radius={R_M_max}")  

M_list = np.array(M_list)
R_list = np.array(R_list)

# Conversion from natural units to km
R_list_converted = R_list * r_conversion_factor

fig = plt.figure()
plt.plot(R_list_converted, M_list)
plt.xlabel("R [km]")
plt.ylabel("M")
plt.title(rf"M-R diagram, $\Gamma$={Gamma}")
plt.grid()
plt.show()


# VARIATION OF k ########################################################################################################

rho_0_c_SI = 5 * 10**14  # [g/cm^3]
rho_0_c = rho_0_c_SI / rho_conversion_factor  
Gamma = 5/2
M_list = []
R_list = []
N = 10
k_list = np.linspace(1700, 5000, N)
for k in k_list:
    M, R = make_star(rho_0_c, Gamma, k)
    M_list.append(M)

fig = plt.figure()
plt.plot(k_list, M_list)
plt.xlabel(r"$\kappa$")
plt.ylabel("M")
plt.title(r"M=M(k) (c=G=solar mass=1)")
plt.grid()
plt.show()


# VARIATION OF Gamma ########################################################################################################
N = 10
rho_0_c_SI = 10**15  # [g/cm^3]
rho_0_c = rho_0_c_SI / rho_conversion_factor 
k = 100
M_list = []
R_list = []
Gamma_list = np.linspace(2.1, 2.5, N)

for Gamma in Gamma_list:
    M, R = make_star(rho_0_c, Gamma, k)
    M_list.append(M)

fig = plt.figure()
plt.plot(Gamma_list, M_list)
plt.xlabel(r"$\Gamma$")
plt.ylabel("M")
plt.title(r"M=M(Gamma)")
plt.grid()
plt.show()
