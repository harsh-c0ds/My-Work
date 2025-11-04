import numpy as np
import matplotlib.pyplot as plt

def eos(P,K,gamma):
    if P <= 0:
        return 0.0, 0.0

    rho_b = (P/K)**(1/gamma) # baryon density
    epsilon = rho_b + P/(gamma - 1) # energy density
    return rho_b, epsilon

def tov_eqs(r,m,P,K,gamma):
    # "Mass and Pressure Equations of State"
    if P <= 0:
        return 0, 0
    rho_b, epsilon = eos(P,K,gamma)
    dm_dr = 4 * np.pi * r**2 * epsilon
    dP_dr = - (rho_b + epsilon) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    return dm_dr, dP_dr

def rk4_step(r, m, P, dr, K, Gamma):
    k1m, k1P = tov_eqs(r, m, P, K, Gamma)

    k2m, k2P = tov_eqs(r + dr/2,
                       m + dr/2 * k1m,
                       P + dr/2 * k1P, K, Gamma)

    k3m, k3P = tov_eqs(r + dr/2,
                       m + dr/2 * k2m,
                       P + dr/2 * k2P, K, Gamma)

    k4m, k4P = tov_eqs(r + dr,
                       m + dr * k3m,
                       P + dr * k3P, K, Gamma)

    m_new = m + dr*(k1m + 2*k2m + 2*k3m + k4m)/6
    P_new = P + dr*(k1P + 2*k2P + 2*k3P + k4P)/6

    return m_new, P_new

def solve_tov(rho_c, K=100, Gamma=2, dr=1e-3):
    P = K * rho_c**Gamma # central pressure
    r = 1e-6 # starting radius
    eps_c, _ = eos(P, K, Gamma) # central energy density
    m = (4.0/3.0) * np.pi * r**3 * eps_c   # initial mass

    Radius = [r]
    Mass = [m]
    while P > 0:
        m, P = rk4_step(r, m, P, dr, K, Gamma)
        r += dr
        Radius.append(r)
        Mass.append(m)

    R = r
    M = m    
    R_iso = R / (1 + M/(2*R))**2
    return M, R, R_iso, np.array(Radius), np.array(Mass)

rho_c = 1.28e-3
M, R, R_iso, Radius, Mass = solve_tov(rho_c)
print(f"Gravitational Mass: {M:.4f}, Radius: {R:.4f}, Isotropic Radius: {R_iso:.4f}")

plt.plot(Radius, Mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.title(r"${M(R)}$ from TOV Solver")
plt.grid()
plt.show()