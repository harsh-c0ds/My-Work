import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from kuibit.simdir import SimDir 
from kuibit.grid_data import UniformGrid  

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

def fx_timeseries(t,x_p,datax,ixd=0, coordinate="x"):     #index value of x as input
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

ixd = 0  # index of the x point for time series
itd = 0  # index of the time point for 1D slice

sim_dir = SimDir("/home/hsolanki/simulations/tov_ET_1/output-0000/tov_ET")

t,x_p,rl,rl_n,datax = get_info("hydrobase","rho",sim_dir,0.0,"x")

print("This works!")