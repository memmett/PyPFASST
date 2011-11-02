Overview
========

PyPFASST is an object-orientated implementation of the PFASST
algorithm.

User interactions with PyPFASST are typically marshaled through the
:py:class:`~pfasst.pfasst.PFASST` class.  This class acts as the
overall controller of the algorithm.  Implementing a PDE solver in
PyPFASST generally consists of the following steps:

#. Instantiate the :py:class:`~pfasst.pfasst.PFASST` controller.

#. Add each level of the grid/solver hierarchy, along with
   appropriate interpolation and restriction operations, to the
   controller.

#. Call the :py:meth:`~pfasst.pfasst.PFASST.run` method of the
   controller.


Levels
------

Each level of the grid hierarchy consists of:

#. A SDC itegrator.  PyPFASST includes two pre-packaged general
   purpose SDC integrators.  They are:

   * :py:class:`~pfasst.explicit.ExplicitSDC` for purely explicit schemes; and

   * :py:class:`~pfasst.imex.IMEXSDC` for implicit/explicit schemes.

   Users are free to provide their own SDC integrators as extensions
   of the :py:class:`~pfasst.sdc.SDC` class.

#. A function evaluator.  These are loosely coupled to the SDC
   integrator used for the level.  For the pre-packaged SDC
   integrators mentioned above, the user should extend either the
   :py:class:`~pfasst.imex.IMEXFEval` class or the
   :py:class:`~pfasst.explicit.ExplicitFEval` class to provide their
   function evaluations.

#. Spatial interpolation and restrction operators.

Levels are added to the controller from finest (level 0) to coarsest
using the controllers :py:meth:`~pfasst.pfasst.PFASST.add_level`
method.  Internally, each level is represented by an instance of the
:py:class:`~pfasst.level.Level` class.


SDC integrators
---------------

Each level has an associated SDC integrator.  The SDC integrators are
implemented as extensions of the base :py:class:`~pfasst.sdc.SDC`
class.  User implementations must override the
:py:meth:`~pfasst.sdc.SDC.sweep` method.

The constructor of the base :py:class:`~pfasst.sdc.SDC` class uses the
:py:mod:`~pfasst.mpsdcquad` module to compute SDC integration
matrices.  The base class also provides a
:py:meth:`~pfasst.sdc.SDC.residual` method for computing residuals.


Function evaluators
-------------------

Each level has an associated function evaluator.  The function
evaluators are implemented as extensions of the base
:py:class:`~pfasst.feval.FEval` class, but are more typically
implemented as extensions of either the
:py:class:`~pfasst.imex.IMEXFEval` class or the
:py:class:`~pfasst.explicit.ExplicitFEval` class as dictated by the
SDC integrator.

Each function evaluation class must set its *shape* and *size*
attributes appropriately (see :py:class:`~pfasst.feval.FEval`).


Interpolation and restriction operators
---------------------------------------

XXX: description of how the interpolation and restriction routines are called.


Time interpolation and restriction
----------------------------------

XXX: description of how time interpolation and restriction is done.


Option database
---------------

XXX: description of the option database


Runtime hooks
-------------

XXX: description of hooks, state etc.







