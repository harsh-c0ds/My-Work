# cc -o halo_irecv_send_toggle_3dim_grid_solution.exe halo_irecv_send_toggle_3dim_grid_solution.c MPIX_?[ai]*.c
# qsub -lnodes=8:ppn=24,walltime=0:30:00 ./3d-008nodes-06.sh
date
cd MPI/00
time aprun -n 192 -N 24 ./halo_irecv_send_toggle_3dim_grid_solution.exe < 3d-008nodes-06-inp.txt > 3d-008nodes-06-out.txt
date
