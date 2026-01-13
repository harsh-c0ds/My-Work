import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  
from scipy.ndimage import gaussian_filter1d


def fourier_transform(time_ms, rho, n_freq=3000):


    # Convert to numpy arrays
    t = np.asarray(time_ms)
    rho = np.asarray(rho)

    # Sort by time
    idx = np.argsort(t)
    t = t[idx]
    rho = rho[idx]

    # Remove duplicate time stamps (ESSENTIAL)
    unique_mask = np.diff(t, prepend=t[0] - 1.0) > 0
    t = t[unique_mask]
    rho = rho[unique_mask]

    # Time differences
    dt_all = np.diff(t)
    dt_min = np.min(dt_all[dt_all > 0])

    # Total duration (ms)
    T = t[-1] - t[0]

    # Frequency grid in kHz (1/ms)
    f_min = 1.0 / T
    f_max = 0.5 / dt_min
    freq_kHz = np.linspace(1, 9, n_freq)

    # Trapezoidal integration weights (ms)
    dt = np.zeros_like(t)
    dt[1:-1] = 0.5 * (t[2:] - t[:-2])
    dt[0] = t[1] - t[0]
    dt[-1] = t[-1] - t[-2]

    # Fourier transform (integral definition)
    rho_tilde = np.array([
        np.sum(rho * np.exp(-2j * np.pi * f * t) * dt)
        for f in freq_kHz
    ])

    power = np.abs(rho_tilde)**2

    return freq_kHz, power

sim_dir_if = "/home/hsolanki/simulations/tov_IF/output-0000/tov_ET"
sim_dir_p = "/home/hsolanki/simulations/Pol_sim/output-0000/tov_ET"
output_dir = "/home/hsolanki/Programs/My-Work/output/"
file_path = output_dir + "rho_timeseries_P.txt"

filex = "hydrobase-rho.x.asc"
folder = sim_dir_p

print("Looking for files in the folder: {}".format(folder))
os.chdir(folder)
print("Opening file: {}.....".format(filex))
datax = np.loadtxt(filex, comments='#')
print("Reading file....")

# --- pick initial time slice ---
t0 = np.min(datax[:,8])
mask_t0 = datax[:,8] == t0
data_t0 = datax[mask_t0]

# --- extract x and rho ---
x   = data_t0[:,9]
rho = data_t0[:,12]

# --- remove atmosphere / vacuum ---
rho_floor = 1e-10   # adjust if needed
mask_star =rho > rho_floor

x_star = x[mask_star]

# --- unique spatial points define ixd ---
x_p = np.unique(x_star)
x_p.sort()

# --- results ---
N_ixd = len(x_p)

print("Number of valid ixd points (center → surface):", N_ixd)
print("ixd range: 0 →", N_ixd-1)
print("Center x ≈", x_p[np.argmin(np.abs(x_p))])
print("Surface x ≈", x_p[-1])


t_s_all = []
rho_all = []


with open(file_path, "r") as f:
    lines = f.readlines()

# Process two lines at a time
for k in range(0, len(lines), 2):
    t_s = np.fromstring(lines[k], sep=" ")
    rho = np.fromstring(lines[k+1], sep=" ")

    t_s_all.append(t_s)
    rho_all.append(rho)

# Convert to object arrays (ragged arrays)
t_s_all = np.array(t_s_all, dtype=object)
rho_all = np.array(rho_all, dtype=object)

print("len(t_s_all): ", len(t_s_all))
print("len(rho_all): ", len(rho_all))  

freq, power = fourier_transform(t_s_all[0], rho_all[0])
power_smooth = gaussian_filter1d(power, sigma=3) # 3
peaks, properties = find_peaks(
    power_smooth,
    prominence=np.max(power) * 0.04,  # stands out from background 0.04
    width=3.5                                   # suppress narrow noise spikes 3.5
)

f_F = freq[peaks[0]]  # Frequency of fundamental mode in kHz

# Remove duplicate time stamps (ESSENTIAL)
unique_mask = np.diff(t_s_all[0], prepend=t_s_all[0][0] - 1.0) > 0
t = t_s_all[0][unique_mask]
rho = rho_all[0][unique_mask]

print(f"1: {len(t_s_all[0])} → 2: {len(t)} after removing duplicates")
dt = np.zeros_like(t)
dt[1:-1] = 0.5 * (t[2:] - t[:-2])
dt[0] = t[1] - t[0]
dt[-1] = t[-1] - t[-2]

rho_tilde = np.sum(
    rho * np.exp(-2j * np.pi * f_F * t) * dt
)
amp_F = abs(rho_tilde)
print(f"F_mode = {f_F} kHz, amp_F = {amp_F}")

F_amp_complex = [rho_tilde]

for i in range(1, 19):

    t_s = t_s_all[i]
    rho = rho_all[i]

    # Remove duplicate / non-increasing times
    unique_mask = np.diff(t_s, prepend=t_s[0] - 1.0) > 0

    t = t_s[unique_mask]
    rho = rho[unique_mask]

    print(f"1: {len(t_s)} → 2: {len(t)} after removing duplicates")

    # Trapezoidal weights
    dt = np.zeros_like(t)

    if len(t) < 2:
        print("WARNING: too few time points, skipping")
        continue

    dt[1:-1] = 0.5 * (t[2:] - t[:-2])
    dt[0] = t[1] - t[0]
    dt[-1] = t[-1] - t[-2]

    # Projection onto F-mode
    rho_tilde_F = np.sum(
        rho * np.exp(-2j * np.pi * f_F * t) * dt
    )

    F_amp_complex.append(rho_tilde_F)
    print(f"amp_F = {abs(rho_tilde_F)}")

F_amp_complex = np.array(F_amp_complex)

# Fix global phase using center point
phase0 = np.angle(F_amp_complex[0])
eig = np.real(F_amp_complex * np.exp(-1j * phase0))

# Normalize (sign preserved)
eig /= np.max(np.abs(eig))

# Radius
r = x_p[:len(eig)]

# Plot
plt.plot(r, eig)
plt.axhline(0, color='k', ls=':')
plt.xlabel("r")
plt.ylabel(r"$|\tilde{\rho}_F(r)|$")
plt.savefig(output_dir + "F_mode_eigenfunction.png")