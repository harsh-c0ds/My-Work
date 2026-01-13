import numpy as np
import matplotlib.pyplot as plt



output_dir = "/home/hsolanki/Programs/My-Work/output/"
output_file = output_dir + "rho_timeseries_P.txt"
data = np.loadtxt(output_file)

t_s_all = data[0::2]   # rows 0,2,4,...
rho_all = data[1::2]   # rows 1,3,5,...

i = 3  # choose index

plt.plot(t_s_all[i], rho_all[i])
plt.xlabel("t_s")
plt.ylabel("rho")
plt.title(f"rho(t) for i = {i}")
plt.savefig(output_dir + "rho_timeseries_i3.png")