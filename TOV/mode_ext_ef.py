import os, sys
from git import Repo
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  
from scipy.ndimage import gaussian_filter1d

def push_to_github(repo_path, commit_message):
    try:
        # Initialize the repo object
        repo = Repo(repo_path)
        
        # Stage all changes (or specify files like output_dir + "*.png")
        repo.git.add(all=True)
        
        # Commit
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{commit_message} - {now}"
        repo.index.commit(full_message)
        
        # Push
        origin = repo.remote(name='origin')
        origin.push(refspec='main:main')
        
        print("Successfully pushed to GitHub!")
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")

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

def detect_star_surface(folder, filename="hydrobase-rho.x.asc", rho_floor=1e-10, time_col=8, x_col=9, rho_col=12, verbose=True):

    if verbose:
        print(f"Looking for files in the folder: {folder}")

    # Preserve caller's working directory
    cwd = os.getcwd()
    try:
        os.chdir(folder)

        if verbose:
            print(f"Opening file: {filename}.....")

        data = np.loadtxt(filename, comments='#')

    finally:
        os.chdir(cwd)

    if verbose:
        print("Reading file....")

    # --- pick initial time slice ---
    t0 = np.min(data[:, time_col])
    mask_t0 = data[:, time_col] == t0
    data_t0 = data[mask_t0]

    # --- extract x and rho ---
    x = data_t0[:, x_col]
    rho = data_t0[:, rho_col]

    # --- remove atmosphere / vacuum ---
    mask_star = rho > rho_floor
    x_star = x[mask_star]
    rho_star = rho[mask_star]  # kept for consistency / future use

    # --- unique spatial points define ixd ---
    x_p = np.unique(x_star)
    x_p.sort()

    N_ixd = len(x_p)
    surface_ixd = N_ixd - 1
    surface_x = x_p[-1]

    if verbose:
        print(f"Number of valid points: {N_ixd}")
        print(f"Physical Surface detected at x ≈ {surface_x:.4f}")

    return x_p, surface_ixd, surface_x

def load_time_series(file_path):
    t_s_all = []
    rho_all = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Process two lines at a time
    for k in range(0, len(lines), 2):
        t_s = np.fromstring(lines[k], sep=" ") ##devide by 203 for lean bssn
        rho = np.fromstring(lines[k+1], sep=" ")

        t_s_all.append(t_s)
        rho_all.append(rho)

    # Convert to object arrays (ragged arrays)
    t_s_all = np.array(t_s_all, dtype=object)
    rho_all = np.array(rho_all, dtype=object)

    print("len(t_s_all): ", len(t_s_all))
    print("len(rho_all): ", len(rho_all))
    return np.array(t_s_all), np.array(rho_all)


sim_dir_if = "/home/hsolanki/simulations/tov_IF/output-0000/tov_ET"
sim_dir_p = "/home/hsolanki/simulations/Pol_sim/output-0000/tov_ET"
sim_dir_lean = "/home/hsolanki/simulations/lean_high/output-0000/tov_ET"
sim_dir_lean_mid = "/home/hsolanki/simulations/lean_bssn_mid/output-0000/tov_ET"
sim_dir_l_beta_1 = "/home/hsolanki/simulations/l_beta_1/output-0000/tov_ET"
output_dir = "/home/hsolanki/Programs/My-Work/output/"
file_path = output_dir + "rho_timeseries_lean_high.txt"
file_path_adm = output_dir + "rho_timeseries_P.txt"

# Detect star surface to get x_p
x_p, surface_ixd_l, surface_x = detect_star_surface(sim_dir_lean, filename="hydrobase-rho.x.asc") 
x_P_P, surface_ixd_p, surface_x_p = detect_star_surface(sim_dir_p, filename="hydrobase-rho.x.asc") 

t_s_all_l, rho_all_l = load_time_series(file_path)
t_s_all_P, rho_all_P = load_time_series(file_path_adm)

t_s_all_l = t_s_all_l / 203  # convert to ms

### Time_series Check ###
ik = 0

t_l = t_s_all_l[ik]
rho_l = rho_all_l[ik]
t_adm = t_s_all_P[ik]
rho_adm = rho_all_P[ik]

lim = np.argmin(t_l <= 5)
lim_adm = np.argmin(t_adm <= 5)

t_l = t_l[:lim]
rho_l = rho_l[:lim]
t_adm = t_adm[:lim_adm]
rho_adm = rho_adm[:lim_adm]

plt.figure(figsize=(10,6))
plt.plot(t_l, rho_l, label="Lean mid", color="blue", alpha=0.6)
plt.plot(t_adm, rho_adm, label="ADM", color="green", alpha=0.6)
plt.xlabel("Time (ms)")
plt.ylabel(r"Central Density $\rho_c$")
plt.title("Timeseries of Central Density Comparison at Centre")
plt.grid(True, linestyle=":")
plt.legend()
plt.savefig(output_dir + "time_series_comparision.png", dpi=300)

# --- Usage at the end of your code ---
# Set the path to the root folder of your local git repository
# my_repo_path = "/home/hsolanki/Programs/My-Work/" 
# push_to_github(my_repo_path, "Updated")




freq_l, power_l = fourier_transform(t_l, rho_l)
freq_adm, power_adm = fourier_transform(t_adm, rho_adm)



#power_smooth = gaussian_filter1d(power, sigma=3) # 3
# peaks_l, properties = find_peaks(
#     power_l,
#     prominence=np.max(power_l) * 0.04,  # stands out from background 0.04
#     width=3.5                               # suppress narrow noise spikes 3.5
# )
# df = freq_l[1] - freq_l[0]  # frequency resolution

peaks_l, properties = find_peaks(
    power_l,
    height=np.max(power_l) * 0.07,
    prominence=np.percentile(power_l, 95) * 0.15,
    width=6,
)

peaks_adm, properties = find_peaks(
    power_adm,
    height=np.max(power_adm) * 0.07,
    prominence=np.percentile(power_adm, 95) * 0.15,
    width=7.5,
)
# peaks_adm, properties = find_peaks(
#     power_adm,
#     prominence=np.max(power_adm) * 0.04,  # stands out from background 0.04
#     width=3.5                                   # suppress narrow noise spikes 3.5
# )

F_c_l = freq_l[peaks_l[0]]  # Frequency of fundamental mode in kHz
F_c_adm = freq_adm[peaks_adm[0]]  # Frequency of fundamental mode in kHz
labels = ["F", "H1", "H2", "H3", "H4", "H5"]

# --- Plotting the Raw Comparison ---
plt.figure(figsize=(10, 6))

# --- plot PSDs ---
plt.plot(freq_l, power_l, alpha=0.6, label="Lean Mid")
plt.plot(freq_adm, power_adm, alpha=0.6, label="ADM")
# --- package peak information ---
spectra = [
    {
        "freq": freq_l,
        "power": power_l,
        "peaks": peaks_l,
        "labels": labels,
        "color": "red",
        "name": "Lean Mid",
        "yoffset": 10
    },
    {
        "freq": freq_adm,
        "power": power_adm,
        "peaks": peaks_adm,
        "labels": labels,
        "color": "purple",
        "name": "ADM",
        "yoffset": -15
    }
]

# --- annotate peaks ---
for spec in spectra:
    for i, peak_idx in enumerate(spec["peaks"][:len(spec["labels"])]):
        x = spec["freq"][peak_idx]
        y = spec["power"][peak_idx]

        plt.axvline(
            x=x,
            color=spec["color"],
            linestyle="--",
            alpha=0.5
        )

        plt.annotate(
            f"{spec['labels'][i]} ({x:.2f} kHz)",
            xy=(x, y),
            xytext=(0, spec["yoffset"]),
            textcoords="offset points",
            ha="center",
            fontstyle="italic",
            color=spec["color"]
        )

plt.xlabel("Frequency (kHz)")
plt.ylabel("Power")
plt.title("Density Power Spectrum with Mode Identification")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig(output_dir + "mode_comparision.png", dpi=300)

# --- Usage at the end of your code ---
# Set the path to the root folder of your local git repository
my_repo_path = "/home/hsolanki/Programs/My-Work/" 
push_to_github(my_repo_path, "Updated")

sys.exit()

# Remove duplicate time stamps (ESSENTIAL)
unique_mask = np.diff(t, prepend=t[0] - 1.0) > 0
t_rm = t[unique_mask]
rho = rho[unique_mask]

print(f"1: {len(t)} → 2: {len(t_rm)} after removing duplicates")
dt = np.zeros_like(t_rm)
dt[1:-1] = 0.5 * (t_rm[2:] - t_rm[:-2])
dt[0] = t_rm[1] - t_rm[0]
dt[-1] = t_rm[-1] - t_rm[-2]

rho_tilde = np.sum(
    rho * np.exp(-2j * np.pi * f_F_c * t_rm) * dt
)
amp_F = abs(rho_tilde)
print(f"F_mode = {f_F_c} kHz, amp_F = {amp_F}")

# plt.figure(figsize=(8,6))
# plt.plot(t, rho)
# plt.xlabel("Time (ms)")
# plt.ylabel(f"rho_c at {ik}")
# plt.savefig(output_dir + "rho_time_series_check.png")



###################### Centre F-Freq Finder ################################

freq, power = fourier_transform(t_s_all[0], rho_all[0])
power_smooth = gaussian_filter1d(power, sigma=3) # 3
peaks, properties = find_peaks(
    power_smooth,
    prominence=np.max(power) * 0.04,  # stands out from background 0.04
    width=3.5                                   # suppress narrow noise spikes 3.5
)

f_F_c = freq[peaks[0]]  # Frequency of fundamental mode in kHz

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
    rho * np.exp(-2j * np.pi * f_F_c * t) * dt
)
amp_F = abs(rho_tilde)
print(f"F_mode = {f_F_c} kHz, amp_F = {amp_F}")

F_amp_complex = []
F_freq = []

for i in range(0, 20):

    t_s = t_s_all[i]
    rho = rho_all[i]

    #### FFT ####

    freq, power = fourier_transform(t_s_all[0], rho_all[0])
    #power_smooth = gaussian_filter1d(power, sigma=3) # 3
    peaks, properties = find_peaks(
    power_smooth,
    prominence=np.max(power) * 0.04,  # stands out from background 0.04
    width=3.5)                                   # suppress narrow noise spikes 3.5

    f_F = freq[peaks[0]]  # Frequency of fundamental mode in kHz
    F_freq.append(f_F)
    print(f"Point {i}: F_mode = {f_F} kHz")

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

# Fix global phase using center point
# phase0 = np.angle(F_amp_complex[0])
# eig = np.real(F_amp_complex * np.exp(-1j * phase0))
eig = np.real(F_amp_complex)
# Normalize (sign preserved)
#eig /= np.max(np.abs(eig))

# Radius
r = x_p[:len(eig)]
r = r*1.477
#print(len(r), len(eig), len(x_p))
# Plot
plt.figure(figsize=(8,6))
plt.plot(r,eig)
plt.xlabel("r (Km)")
plt.ylabel(r"$\tilde{\rho}_F(r)$")
plt.savefig(output_dir + "F_mode_eigenfunction_2.png")

# --- Usage at the end of your code ---
# Set the path to the root folder of your local git repository
my_repo_path = "/home/hsolanki/Programs/My-Work/" 
push_to_github(my_repo_path, "Updated")