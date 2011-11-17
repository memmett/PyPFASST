Tutorial
========


The Advection/Diffusion (AD) example
------------------------------------

To run the AD example::

$ cd PyPFASST
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ cd examples/advection
$ mpirun -n 4 python main.py

This solves the 1d AD equation using a V cycle with 3 PFASST levels
(5, 3, and 2 nodes) and 4 processors.  The logarithm of the maximum
absolute error (compared to an exact solution) is echoed after each
SDC sweep.
