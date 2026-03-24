## Syntax
We use three spaces to mark indents.

# General information
To run python with mpi4py just call as usual `mpirun python3 script.py` (if necessary with --oversubscribe etc.).

Note: Run our code with python 3.5 or above only since some functions, e.g. input(), have changed significantly between Python2 and Python3!

mpi4py provides Python bindings for the Message Passing Interface (MPI) standard. It is implemented on top of the MPI-1/2/3 specification and exposes an API which grounds on the standard MPI-2 C++ bindings.
Dependencies

    - Python 2.7, 3.5 or above, or PyPy 2.0 or above.
    - A functional MPI 1.x/2.x/3.x implementation like MPICH or Open MPI built with shared/dynamic libraries.

Pay attention to the difference between upper and lower case calls, see https://mpi4py.readthedocs.io/en/stable/tutorial.html.

Module functions MPI.Init() or MPI.Init_thread() and MPI.Finalize() provide MPI initialization and finalization respectively. Module functions MPI.Is_initialized() and MPI.Is_finalized() provide the respective tests for initialization and finalization (https://mpi4py.readthedocs.io/en/stable/overview.html#environmental-management).

Note that

- MPI_Init() or MPI_Init_thread() is actually called when you import the MPI module from the mpi4py package, but only if MPI is not already initialized. In such case, calling MPI.Init() or MPI.Init_thread() from Python is expected to generate an MPI error, and in turn an exception will be raised.

- MPI_Finalize() is registered (by using Python C/API function Py_AtExit()) for being automatically called when Python processes exit, but only if mpi4py actually initialized MPI. Therefore, there is no need to call MPI.Finalize() from Python to ensure MPI finalization.

# Standard error handling
See https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html. At import time, mpi4py initializes the MPI execution environment calling MPI_Init_thread() and installs an exit hook to automatically call MPI_Finalize() just before the Python process terminates. Additionally, mpi4py overrides the default MPI.ERRORS_ARE_FATAL error handler in favor of MPI.ERRORS_RETURN, which allows translating MPI errors in Python exceptions. These departures from standard MPI behavior may be controversial, but are quite convenient within the highly dynamic Python programming environment. Third-party code using mpi4py can just from mpi4py import MPI and perform MPI calls without the tedious initialization/finalization handling. MPI errors, once translated automatically to Python exceptions, can be dealt with the common try…except…finally clauses; unhandled MPI exceptions will print a traceback which helps in locating problems in source code.

Unfortunately, the interplay of automatic MPI finalization and unhandled exceptions may lead to deadlocks. In unattended runs, these deadlocks will drain the battery of your laptop, or burn precious allocation hours in your supercomputing facility.


# API reference
https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.html#module-mpi4py.MPI or https://mpi4py.github.io/apiref/index.html

# Communication of buffer-like objects #

You have to use method names starting with an upper-case letter (of the Comm class), like Send(), Recv(), Bcast(), Scatter(), Gather().

In general, buffer arguments to these calls must be explicitly specified by using a 2/3-list/tuple like [data, MPI.DOUBLE], or [data, count, MPI.DOUBLE] (the former one uses the byte-size of data and the extent of the MPI datatype to define count).

For vector collectives communication operations like Scatterv() and Gatherv(), buffer arguments are specified as [data, count, displ, datatype], where count and displ are sequences of integral values.

Automatic MPI datatype discovery for NumPy arrays and PEP-3118 buffers is supported, but limited to basic C types (all C/C99-native signed/unsigned integral types and single/double precision real/complex floating types) and availability of matching datatypes in the underlying MPI implementation. In this case, the buffer-provider object can be passed directly as a buffer argument, the count and MPI datatype will be inferred.

[[https://mpi4py.readthedocs.io/en/stable/tutorial.html]]

