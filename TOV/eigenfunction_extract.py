import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

### Eigenfunction Extraction ###

#     #Detect star surface to get x_p
# x_p, surface_ixd_l, surface_x = detect_star_surface(sim_dir_p, filename="hydrobase-rho.x.asc") 


F_amp_complex = []
F_freq = []

for i in range(0, 45):

    t_s = t_s_all_l[i]
    rho = rho_all_l[i]
    lim = (t_s >= 0.5) & (t_s <= 5)
    t_s = t_s[lim]
    rho = rho[lim]

    #### FFT ####

    freq, power = fourier_transform(t_s, rho)
    peaks, properties = find_peaks(
        power,
        height=np.max(power) * 0.07,
        prominence=np.percentile(power, 95) * 0.35,
        width=8,
    )                                # suppress narrow noise spikes 3.5

    f_F = freq[peaks[1]]  # Frequency of fundamental mode in kHz
    F_freq.append(f_F)
    print(f"Point {i}: F_mode = {f_F} kHz")
    #f_F = F_c_l

    # Remove duplicate / non-increasing times
    unique_mask = np.diff(t_s, prepend=t_s[0] - 1.0) > 0

    t = t_s[unique_mask]
    rho = rho[unique_mask]

    print(f"1: {len(t_s)} → 2: {len(t)} after removing duplicates")

    # Trapezoidal weights
    dt = np.zeros_like(t)
    dt[1:-1] = 0.5 * (t[2:] - t[:-2])
    dt[0] = t[1] - t[0]
    dt[-1] = t[-1] - t[-2]

    # Projection onto F-mode
    rho_tilde_F = np.sum(
        rho * np.exp(-2j * np.pi * f_F * t) * dt
    )

    F_amp_complex.append(rho_tilde_F)
    #print(f"amp_F = {abs(rho_tilde_F)}")

F_amp_complex = np.array(F_amp_complex)


eig = np.real(F_amp_complex)


# Radius
r = x_p[:len(eig)]
r = r*1.477

# Plot
plt.figure(figsize=(8,6))
plt.plot(r,eig)
plt.xlabel("r (Km)")
plt.ylabel(r"$\tilde{\rho}_F(r)$")