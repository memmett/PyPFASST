Tutorial
========


Advection/Diffusion
-------------------

To run the advection/diffustion (AD) example::

$ cd PyPFASST
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ cd examples/advection
$ mpirun -n 4 python main.py

This solves the 1d AD equation using a V cycle with 3 PFASST levels
(5, 3, and 2 nodes) and 4 processors.  The logarithm of the maximum
absolute error (compared to an exact solution) is echoed after each
SDC sweep.

Please skim over the :doc:`overview <overview>` documentation to get a
jist of how PyPFASST is used, then consider the ``main.py`` script of
the AD example:

.. literalinclude:: ../examples/advection/main.py
   :language: python
   :linenos:

On line 38 we import the ``AD`` class and the spatial ``interpolate``
and ``restrict`` functions for this example from the ``ad.py`` file:

.. literalinclude:: ../examples/advection/ad.py
   :language: python
   :linenos:
