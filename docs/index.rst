PFASST
======

PFASST is an algorithm for parallelizing PDEs in time.  The PFASST
acronym stands for **Parallel Full Approximation Scheme in Space
and Time**.

Parallelizing *N+1* dimensional PDEs (*N* spatial dimensions and one
time dimension) is typically done by **decomposing space**.  That is,
the spatial domain of the PDE is broken down into several pieces.
Once this is done, the spatial operators of the PDE can be evaluated
on each piece of the domain concurrently (ie, in parallel) at each
time step.

Another way of parallelizing PDEs is by **decomposing time**.

To efficiently parallelize PDEs in time, PFASST brings together
several different numerical methods in a novel way.  These
include:

* Parareal: time decomposition for ODEs;

* Full Approximation Scheme (FAS): multigrid corrections for
  non-linear operators; and

* Spectral Deferred Corrections (SDC): an iterative time-stepping
  technique with arbitrarly high order accuracy.

In practice, temporal parallelization is only attractive if the
parallel efficiency is greater than (additional) spatial
parallelization.  If, for example,

* you have more processors than you can shake a stick at;

* you have saturated your spatially-parallel algorithm, but need it to
  go even faster; or

* have a problem in which the physics can modeled with various sets of
  equations with increasing computational complexity; then

temporal paralellization might be attractive.  Check out the `PFASST
gallery`_ to see some problems that PFASST has been applied to.

The maths behind the PFASST algorithm are described on the
:doc:`maths` page.

This work was supported by the Director, DOE Office of Science, Office
of Advanced Scientific Computing Research, Office of Mathematics,
Information, and Computational Sciences, Applied Mathematical Sciences
Program, under contract DE-SC0004011.  This work is currently authored
by `Michael L. Minion`_ and `Matthew Emmett`_.  Contributions are
welcome -- please contact `Matthew Emmett`_.

This website is a work in progress!


PyPFASST
========

PyPFASST is a Python implementation of the PFASST algorithm.  This
allows scientists to develop their code in a high-level language
without sacrificing efficiency, as existing Fortran, C, and C++ codes
can be called from within Python.

**Main parts of the documentation**

* :doc:`Download <download>` - download and installation instructions.
* :doc:`Tutorial <tutorial>` - getting started and basic usage.
* :doc:`Overview <overview>` - design and interface overview.
* :doc:`Reference <reference>` - reference and API documentation.


.. toctree::
   :hidden:

   self
   maths
   download
   tutorial
   overview
   reference


Contributing
------------

PyPFASST is released under the simplified BSD license, and is free for
all uses as long as the copyright notices remain intact.

If you would like to contribute to the development of PyPFASST,
please, dive right in by visiting the `PyPFASST project page`_.


.. _`Michael L. Minion`: http://amath.unc.edu/Minion/Minion
.. _`Matthew Emmett`: http://www.unc.edu/~mwemmett/
.. _`PyPFASST project page`: https://github.com/memmett/PyPFASST
.. _`PFASST gallery`: http://www.unc.edu/~mwemmett/pfasst_gallery.html
