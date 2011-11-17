Reference
=========

Throughout the PyPFASST documentation, *F* and *G* refer to fine and
coarse levels.


Top-level classes
-----------------

PFASST class
^^^^^^^^^^^^

.. autoclass:: pfasst.pfasst.PFASST
   :members: run, add_level, add_hook, simple_communicators


FEval class
^^^^^^^^^^^

.. autoclass:: pfasst.feval.FEval
   :members:


SDC class
^^^^^^^^^

.. autoclass:: pfasst.sdc.SDC
   :members:


Level class
^^^^^^^^^^^

.. autoclass:: pfasst.level.Level
   :members:


Helpers
-------


IMEX
^^^^

.. automodule:: pfasst.imex
   :members:

Explicit
^^^^^^^^

.. automodule:: pfasst.explicit
   :members:

MPI
^^^

.. autoclass:: pfasst.mpi.PFASSTMPI
   :members:


.. Options
.. ^^^^^^^

.. .. automodule:: pfasst.options
..    :members:


Low-level routines (guts)
-------------------------

Interpolate
^^^^^^^^^^^

.. automodule:: pfasst.interpolate
   :members:

Restrict
^^^^^^^^

.. automodule:: pfasst.restrict
   :members:

State
^^^^^

.. autoclass:: pfasst.state.State
   :members:

FAS
^^^

.. automodule:: pfasst.fas
   :members:


Runners
^^^^^^^

.. automodule:: pfasst.runner
   :members:

.. automodule:: pfasst.serial
   :members:

.. automodule:: pfasst.parallel
   :members:
