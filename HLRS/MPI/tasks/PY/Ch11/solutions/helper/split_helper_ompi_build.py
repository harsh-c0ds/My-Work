from cffi import FFI

ffi = FFI()
ffi.set_source("split_helper_ompi", r"#include <mpi.h>")
ffi.cdef(
   r"""
   const int OMPI_COMM_TYPE_NODE;
   const int MPI_COMM_TYPE_SHARED;
   const int OMPI_COMM_TYPE_HWTHREAD;
   const int OMPI_COMM_TYPE_CORE;
   const int OMPI_COMM_TYPE_L1CACHE;
   const int OMPI_COMM_TYPE_L2CACHE;
   const int OMPI_COMM_TYPE_L3CACHE;
   const int OMPI_COMM_TYPE_SOCKET;
   const int OMPI_COMM_TYPE_NUMA;
   const int OMPI_COMM_TYPE_BOARD;
   const int OMPI_COMM_TYPE_HOST;
   const int OMPI_COMM_TYPE_CU;
   const int OMPI_COMM_TYPE_CLUSTER;
   """
)

if __name__ == "__main__":
   ffi.compile(verbose=True)
