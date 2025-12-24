import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from kuibit.simdir import SimDir 
from kuibit.grid_data import UniformGrid
from astropy.timeseries import LombScargle  

################################################
 # Define constants and conversion factors
################################################
# constants, in SI
G = 6.673e-11       # m^3/(kg s^2)
c = 299792458       # m/s
M_sol = 1.98892e30  # kg
# convertion factors
M_to_ms = 1./(1000*M_sol*G/(c*c*c))
M_to_density = c**5 / (G**3 * M_sol**2) # kg/m^3

def get_info(thorn,quantity, folder,t0,coordinate="x"):
        print("Looking for files in the folder: {}".format(folder))
        os.chdir(folder)
        filex = f"{thorn}-{quantity}.{coordinate}.asc"
        print("Opening file: {}.....".format(filex))
        datax = np.loadtxt(filex, comments='#')
        print("Reading file....")
        
        # Number of iterations
        it = np.unique(datax[:, 0])
        it_n = len(it)
        print("Number of iterations:", it_n)
        
        # Time values
        t = np.unique(datax[:, 8])
        # X values
        if coordinate == "x": 
           x_p = np.unique(datax[:, 9])
        if coordinate == "y": 
           x_p = np.unique(datax[:, 10])
        if coordinate == "z": 
           x_p = np.unique(datax[:, 11])
         
        # Refinement levels	
        rl = np.unique(datax[:, 2])
        print('N points in x_p:')
        print(len(x_p))
        rl_n = len(rl)
        print("Total number of refinement levels:", rl_n)

        if t0<t[-1] and t0!=0:
            t=t[t>t0]
            t_n = len(t)
            print("Number of different time values:", t_n)

        # Points
            x_p_n = len(x_p)
            print("Total number of points:", x_p_n)


            points_per_rl = []
            rl_max_point = []
            for i in range(rl_n):
                x_in_rl = np.unique(datax[datax[:, 2] == rl[i], 9])
                points_in_rl = len(x_in_rl)
                print("Number of points in refinement level", i, ":", points_in_rl)
                rl_max_point.append(np.max(x_in_rl))
                points_per_rl.append(points_in_rl)
       # rl_max_point.append(0.0)
        
        return t,x_p,rl,rl_n,datax

def fx_timeseries(t,x_p,datax,ixd, coordinate="x"):     #index value of x as input
    #create output lists
    print(os.getcwd())
    t_n = len(t)
    time_values = []
    f_xt_values = []
    #print(f"Calculating timeseries for {coordinate} = {x_p[ixd]}")
    print(f"Starting at  t = {t[0]}")
 # create filter for time steps
    for j in range(t_n): 
        t_index = datax[:,8] == t[j]
# get data  as t,coordinate,f(t,coordinate) 
        if coordinate == "x": 
          f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,9]  , datax[t_index,12]  ))
        if coordinate == "y": 
          f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,10]  , datax[t_index,12]  ))
        if coordinate == "z": 
          f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,11]  , datax[t_index,12]  ))
#now x=f_x_ti[0][:] and f(x)=f_x_ti[1][:]
 #create filter for space points
        if ixd==0:
           x_index = f_x_ti[1][:] == 0.0
        else:
           x_index = f_x_ti[1][:] == x_p[ixd]

 # save t, x and f(x,t) in a list (use lists to improve efficiency when extending)
        tj = (f_x_ti[0][x_index]).tolist()
        f_xi_tj = (f_x_ti[2][x_index]).tolist()

 #append values
        time_values.extend(tj)
        f_xt_values.extend(f_xi_tj)
        if(j==np.round(1/8*t_n) or j==np.round(1/4*t_n)) or j==np.round(3/8*t_n) or  j==np.round(1/2*t_n) or j==np.round(5/8*t_n) or j==np.round(3/4*t_n) or j==np.round(7/8*t_n):
                print("Progress: {} %".format(j/t_n *100))
    print("Done...!")
    return time_values,f_xt_values

def get_1d_slice(tk1, xk1, datax, itd, coordinate):

    #print(f"Getting 1d-{coordinate} slice at t = {tk1[itd]}")

    t_index = datax[:,8] == tk1[itd] # get all values at fixed time t_i = tk1[itd]

       # get data  as t,coordinate,f(t,coordinate) 
    if coordinate == "x": 
       f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,9]  , datax[t_index,12]  ))
    if coordinate == "y": 
       f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,10]  , datax[t_index,12]  ))
    if coordinate == "z": 
       f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,11]  , datax[t_index,12]  ))

       # split into t_i, x_j, f(x_j,t_i) 
    tj = (f_x_ti[0]).tolist()    # t_i should be all the same
    xj = (f_x_ti[1]).tolist()    # array of {x,y,z} values
    f_xi_tj = (f_x_ti[2]).tolist()

    # Convert lists back to numpy arrays for sorting
    xj = np.array(xj)
    f_xi_tj = np.array(f_xi_tj)
    # Sort the arrays based on xj
    sorted_indices = np.argsort(xj)
    # Reorder both xj and f_xi_tj based on sorted indices
    xj_sorted = xj[sorted_indices]
    f_xi_tj_sorted = f_xi_tj[sorted_indices]

    return xj_sorted, f_xi_tj_sorted

sim_dir_if = "/home/hsolanki/simulations/tov_IF/output-0000/tov_ET"
sim_dir_p = "/home/hsolanki/simulations/tov_ET_1/output-0000/tov_ET"
output_dir = "/home/hsolanki/Programs/My-Work/output/"

#### figure out the total number of ixd ####

filex = "hydrobase-rho.x.asc"
folder = sim_dir_if

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

sys.exit()

ixd = 0  # index of the x point for time series
itd = 0  # index of the time point for 1D slice



sim = "if"  # "if", "p", "both"

if sim == "if":
   t_if,x_p_if,rl_if,rl_n_if,datax_if = get_info("hydrobase","rho",sim_dir_if,0.0,"x")
   time_values_if,f_xt_values_if = fx_timeseries(t_if,x_p_if,datax_if,ixd==10,"x")

   time_values_if = np.array(time_values_if)/203  # convert to ms
   rho_ts_if = (np.array(f_xt_values_if) - f_xt_values_if[0]) /f_xt_values_if[0]  # normalize density
   idxx = np.argmax(time_values_if >= 4)
   time_values_if = time_values_if[:idxx]
   rho_ts_if = rho_ts_if[:idxx]

   t_s = time_values_if  # ms
   frequency = np.linspace(1, 9, 5000)  # 0–9 kHz
   power = LombScargle(t_s, rho_ts_if).power(frequency)

   plt.figure(figsize=(8,6))
   plt.plot(frequency, power, color="red", linewidth=1.5, label="Ideal Fluid EOS")
   plt.xlabel("Frequency (kHz)")
   plt.ylabel(r"${|\rho/\rho_{c,0}|}^2$")
   plt.title(r"FFT of Density")
   plt.grid(True, linestyle=":")
   plt.legend()
   plt.savefig(output_dir + "fft_density_IF.png", dpi=300)


   plt.figure(figsize=(8,6))
   plt.plot(time_values_if, rho_ts_if, color="red", linewidth=1.5, label="Ideal Fluid EOS")
   plt.xlabel("Time (ms)")
   plt.ylabel(r"$\rho/\rho_{c,0}$")
   plt.title(r"Timeseries of Density")
   plt.grid(True, linestyle=":")
   plt.legend()
   plt.savefig(output_dir + "time_series_density_IF.png", dpi=300)

elif sim == "p":
   t_p,x_p_p,rl_p,rl_n_p,datax_p = get_info("hydrobase","rho",sim_dir_p,0.0,"x")
   time_values_p,f_xt_values_p = fx_timeseries(t_p,x_p_p,datax_p,ixd==10,"x")

   time_values_p = np.array(time_values_p)/203  # convert to ms
   rho_ts_p = (np.array(f_xt_values_p) - f_xt_values_p[0])/f_xt_values_p[0]  # normalize density
   idxx = np.argmax(time_values_p >= 4)
   time_values_p = time_values_p[:idxx]
   rho_ts_p = rho_ts_p[:idxx]

   t_s = time_values_p  # ms
   frequency = np.linspace(1, 9, 5000)  # 0–9 kHz
   power = LombScargle(t_s, rho_ts_p).power(frequency)

   plt.figure(figsize=(8,6))
   plt.plot(frequency, power, color="blue", linewidth=1.5, label="Polytropic EOS")
   plt.xlabel("Frequency (kHz)")
   plt.ylabel(r"${|\rho/\rho_{c,0}|}^2$")
   plt.title(r"FFT of Density")
   plt.grid(True, linestyle=":")
   plt.legend()
   plt.savefig(output_dir + "fft_density_P.png", dpi=300)

elif sim == "both":
   t_if,x_p_if,rl_if,rl_n_if,datax_if = get_info("hydrobase","rho",sim_dir_if,0.0,"x")
   time_values_if,f_xt_values_if = fx_timeseries(t_if,x_p_if,datax_if,ixd==10,"x")

   time_values_if = np.array(time_values_if)/203  # convert to ms
   rho_ts_if = (np.array(f_xt_values_if) - f_xt_values_if[0]) /f_xt_values_if[0]  # normalize density
   idxx = np.argmax(time_values_if >= 7)
   time_values_if = time_values_if[:idxx]
   rho_ts_if = rho_ts_if[:idxx]

   t_p,x_p_p,rl_p,rl_n_p,datax_p = get_info("hydrobase","rho",sim_dir_p,0.0,"x")
   time_values_p,f_xt_values_p = fx_timeseries(t_p,x_p_p,datax_p,ixd==10,"x")

   time_values_p = np.array(time_values_p)/203  # convert to ms
   rho_ts_p = (np.array(f_xt_values_p) - f_xt_values_p[0])/f_xt_values_p[0]  # normalize density
   idxx = np.argmax(time_values_p >= 4)
   time_values_p = time_values_p[:idxx]
   rho_ts_p = rho_ts_p[:idxx]
   rho_ts_p = rho_ts_p - np.mean(rho_ts_p)

   t_s_if = time_values_if  # ms
   frequency_if = np.linspace(1, 9, 5000)  # 0–9 kHz
   power_if = LombScargle(t_s_if, rho_ts_if).power(frequency_if)

   t_s_p = time_values_p  # ms
   frequency_p = np.linspace(1, 9, 5000)  # 0–9 kHz
   power_p = LombScargle(t_s_p, rho_ts_p).power(frequency_p)

   plt.figure(figsize=(8,6))
   plt.plot(frequency_if, power_if, color="red", linewidth=1.5, label="Ideal Fluid EOS")
   plt.plot(frequency_p, power_p, color="blue", linewidth=1.5, label="Polytropic EOS")
   plt.xlabel("Frequency (kHz)")
   plt.ylabel(r"${|\rho/\rho_{c,0}|}^2$")
   plt.title(r"FFT of Density")
   plt.grid(True, linestyle=":")
   plt.legend()
   plt.savefig(output_dir + "fft_density_comparison.png", dpi=300)


sys.exit()

###### Density and Lapse 1D Slice ########

#t_p,x_p_p,rl_p,rl_n_p,datax_p = get_info("hydrobase","rho",sim_dir_p,0.0,"x")
# t_2,x_p_2,rl_2,rl_n_2,datax_2 = get_info("admbase","lapse",sim_dir,0.0,"x")

# xj_sorted_1, rho = get_1d_slice(t_1, x_p_1, datax_1, itd, "x")
# xj_sorted_2, lapse = get_1d_slice(t_2, x_p_2, datax_2, itd, "x")

# xj_11, rho_1 = get_1d_slice(t_1, x_p_1, datax_1, 10, "x")
# xj_21, lapse_1 = get_1d_slice(t_2, x_p_2, datax_2, 10, "x")

# time = 1250/204

# rho = rho/rho[0]  # normalize density

# xj_sorted_1 = xj_sorted_1 * 1.477  # convert to km
# xj_sorted_2 = xj_sorted_2 * 1.477  # convert to km
# xj_11 = xj_11 * 1.477  # convert to km
# xj_21 = xj_21 * 1.477  # convert to km

# idx = np.argmax(xj_sorted_1 >= 15)
# xj_sorted_1 = xj_sorted_1[:idx]
# xj_sorted_2 = xj_sorted_2[:idx]
# rho = rho[:idx]
# lapse = lapse[:idx]
# xj_11 = xj_11[:idx]
# xj_21 = xj_21[:idx]
# rho_1 = rho_1[:idx]
# lapse_1 = lapse_1[:idx]

# plt.figure(figsize=(8,6))

# # --- Left axis (density ρ) ---
# ax1 = plt.gca()
# ax1.set_xlabel("x (km)")
# ax1.set_ylabel(r"$\rho/\rho_{c,0}$")
# ax1.set_ylim(0, 1.2)   # match your figure

# # Plot density
# ax1.plot(xj_sorted_1, rho, color="black", linewidth=1.5, label="ρ, t = 0")
# ax1.plot(xj_11, rho_1, color="black", linestyle="--", linewidth=1.5,
#          label="ρ, t = {:.3f} ms".format(time))
# ax1.xaxis.set_major_locator(plt.AutoLocator())
# ax1.yaxis.set_major_locator(plt.AutoLocator())
# ax1.minorticks_on()

# # --- Right axis (lapse α) ---
# ax2 = ax1.twinx()
# ax2.set_ylabel(r"$\alpha$")
# ax2.set_ylim(0.60, 0.90)   # right-axis limits from your paper

# # Plot lapse
# ax2.plot(xj_sorted_2, lapse, color="gray", linewidth=1.5, label="α, t = 0")
# ax2.plot(xj_21, lapse_1, color="gray", linestyle="--", linewidth=1.5,
#          label="α, t = {:.3f} ms".format(time))
# ax2.xaxis.set_major_locator(plt.AutoLocator())
# ax2.yaxis.set_major_locator(plt.AutoLocator())
# ax2.minorticks_on()
# # --- Add grid, legend, etc. ---
# ax1.grid(True, linestyle=":")
# plt.title("1D Slice of Density and Lapse")

# # Combine legends from both axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# plt.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
# plt.savefig(output_dir + "density_lapse.png", dpi=300)



# rho_ts_if = rho_ts_if - np.mean(rho_ts_if)





# ax = plt.gca()

# plt.figure(figsize=(8,6))
# plt.plot(time_values, rho_ts, color="blue", linewidth=1.5)
# plt.xlabel("Time (ms)")
# plt.ylabel(r"$\rho/\rho_{c,0}$")
# plt.title(r"Time Series of Density")
# plt.grid(True, linestyle=":")
# ax.xaxis.set_major_locator(plt.AutoLocator())
# ax.yaxis.set_major_locator(plt.AutoLocator())
# ax.minorticks_on()
# plt.savefig(output_dir + "density_timeseries.png", dpi=300)


####### Power Spectrum Calculation ########

t_s = time_values_if  # ms
frequency = np.linspace(1, 9, 5000)  # 0–9 kHz
power = LombScargle(t_s, rho_ts_if).power(frequency)


# order = np.argsort(time_ms)
# time_ms = time_ms[order]
# rho_ts = rho_ts[order]

# t_unique, inv = np.unique(time_ms, return_inverse=True)
# rho_ts = np.array([rho_ts[inv == i].mean() for i in range(len(t_unique))])
# time_ms = t_unique

# dt = np.diff(time_ms)
# mask = dt > 1e-6
# time_ms = time_ms[np.insert(mask, 0, True)]
# rho_ts = rho_ts[np.insert(mask, 0, True)]

# rho_ts = rho_ts / rho_ts[0]
# rho_ts -= np.mean(rho_ts)
# rho_ts *= np.hanning(len(rho_ts))

# dt_eff = np.median(np.diff(time_ms))
# frequency = np.linspace(0.1, 0.5 / dt_eff, 5000)

# power = LombScargle(time_ms, rho_ts, center_data=False, fit_mean=False).power(frequency)



#print("dt min / median / max =", np.min(dt), np.median(dt), np.max(dt))
# print(f"dt = {time_values[11]-time_values[10]} ms")
# print(f"Number of time samples = {len(rho_ts)}")
# plt.figure(figsize=(8,6))
# plt.plot(frequency, power, color="red", linewidth=1.5)
# plt.xlabel("Frequency (kHz)")
# plt.ylabel("Power")
# plt.title("Power Spectrum of Density Time Series")
# plt.grid(True, linestyle=":")
# plt.savefig(output_dir + "power_fft.png", dpi=300)


###### Radial Velocity FFT ########

# ixd = 0
# t,x_p,rl,rl_n,datax = get_info("hydrobase","vel",sim_dir,0.0,"x")
# time_values_vel,vel_values = fx_timeseries(t,x_p,datax,ixd,"x")

# time_values_vel = np.array(time_values_vel)/203  # convert to ms
# vel_values = np.array(vel_values)  # in units of c

# frequency_vel = np.linspace(0.01, 9, 5000)  # 0–9 kHz
# vel = LombScargle(time_values_vel, vel_values).power(frequency_vel)

# plt.figure(figsize=(8,6))
# plt.plot(frequency_vel, vel, color="red", linewidth=1.5)
# plt.xlabel("Frequency (kHz)")
# plt.ylabel("Radial Velocity")
# plt.title("Spectrum Velocity Time Series")
# plt.grid(True, linestyle=":")
# plt.savefig(output_dir + "velocity_spectrum.png", dpi=300)