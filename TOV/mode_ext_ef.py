import numpy as np
import matplotlib.pyplot as plt

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

output_dir = "/home/hsolanki/Programs/My-Work/output/"
file_path = output_dir + "rho_timeseries_P.txt"
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

plt.plot(t_s_all[13], rho_all[13])
plt.xlabel("Time (s)")
plt.ylabel("Density perturbation")
plt.title("Density perturbation vs Time for first radial point")
plt.savefig(output_dir + "time_series_try.png")
