import numpy as np
import matplotlib.pyplot as plt



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

i = 3  # choose index

plt.plot(t_s_all[i], rho_all[i])
plt.xlabel("t_s")
plt.ylabel("rho")
plt.title(f"rho(t) for i = {i}")
plt.savefig(output_dir + "rho_timeseries_i3.png")