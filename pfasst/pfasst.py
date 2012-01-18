"""PyPFASST PFASST class."""

# Copyright (c) 2011, Matthew Emmett.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from mpi4py import MPI

import mpi
import state
import serial
import parallel
import rk

from level import Level


class PFASST(object):
  """Main PFASST class/driver.

  Attributes:

  .. attribute:: levels

     Array of PFASST levels.  See :meth:`~pfasst.pfasst.PFASST.add_level`.

  .. attribute:: mpi

     PFASST MPI helper class.  See :class:`pfasst.mpi.PFASSTMPI`.

  .. attribute:: state

     PFASST state class.  See :class:`pfasst.state.State`.


  Each PFASST level contains the following state arrays:

  .. attribute:: q0

     Array of initial conditions.

  .. attribute:: qend

     Array of final values.

  .. attribute:: tau

     Array of FAS corrections.

  .. attribute:: qSDC

     Array of unknowns at each SDC node.

  .. attribute:: fSDC

     Array of function values at each SDC node.


  Each PFASST level also has the following attributes:

  .. attribute:: feval

     Function evaluator (see :class:`~pfasst.feval.FEval`).

  .. attribute:: sdc

     SDC integrator (see :class:`~pfasst.sdc.SDC`).

  .. attribute:: interpolate

     Interpolator (see below).

  .. attribute:: restrict

     Restrictor (see below).


  The *state* attribute has the following attributes:

  .. attribute:: cycle

     Current position in the PFASST multi-level cycle.

  .. attribute:: iteration

     Current PFASST iteration.

  .. attribute:: predictor

     True if we're currently in the predictor.


  Keep in mind that:

  * Level 0 is considered the *finest* level.

  * The interpolate routine at level *N* (= *iF*) is called to
    interpolate from level *N+1* (= *iG*) to *N*.  For example::

    >>> levels[iF].interpolate(yF, yG,
    >>>                        fevalF=levels[iF].feval,
    >>>                        fevalG=levels[iG].feval)

    interpolates the coarse y values in *yG* at level *iG* to level
    *iF* and stores the result in *yF*.

  * The restrict routine at level *N* is called to restrict from
    *level N* to level *N+1*.  For example::

    >>> levels[iF].restrict(yF, yG,
    >>>                     fevalF=levels[iF].feval,
    >>>                     fevalG=levels[iG].feval)

    restricts the fine y values in *yF* at level *iF* to level *iG*
    and stores the result in *yG*.

  * PFASST takes care of computing increments and calling your
    interpolation routines to interpolate increments.  **Your
    interpolation routines don't need to compute increments.**

  """

  def __init__(self):

    self.levels = []                 # array of levels
    self.state  = state.State()      # state
    self.mpi    = mpi.PFASSTMPI()    # mpi

    self.state.mpi   = self.mpi


  #############################################################################

  def simple_communicators(self, nspace=1, ntime=MPI.COMM_WORLD.size,
                           comm=MPI.COMM_WORLD, **kwargs):
    """Set the MPI communicators using simple colouring."""

    self.mpi.create_simple_communicators(nspace, ntime, comm=comm)


  #############################################################################

  def add_level(self, feval, sdc, interpolate=None, restrict=None):
    """Add a level to the PFASST hierarchy.

    :param feval: implicit/explicit function evaluator (instance
            of :class:`pfasst.feval.FEval`)
    :param sdc: implicit/explicit SDC sweeper (instance
            of :class:`pfasst.sdc.SDC`)
    :param interpolate: interpolate from coarser level to this
            level (callable)
    :param restrict: restrict from this level to coarser level
            (callable)

    Levels should be added from finest (level 0) to coarest.

    The *interpolate* callable is called as::

    >>> interpolate(yF, yG, fevalF=fevaF, fevalG=fevalG, **kwargs)

    and should interpolate the coarse y values *yG* that
    correspond to the coarse evaluator *fevalG* to the fine y
    values *yF* that correspond to the fine evaluator *fevalF*.
    The (flattened) result should be stored in *yF*.

    The *restrict* callable is called as::

    >>> restrict(yF, yG, fevalF=fevalF, fevalG=fevalG, **kwargs)

    and should restrict the fine y values *yF* that correspond to
    the fine evaluator *fevalF* to the coarse y values *yG* that
    correspond to the coarse evaluator *fevalG*.  The (flattened)
    result should be stored in *yG*.

    """

    level = Level()

    level.feval       = feval
    level.sdc         = sdc
    level.interpolate = interpolate
    level.restrict    = restrict

    level.level       = len(self.levels)
    level.mpi         = self.mpi
    level.state       = self.state
    level.hooks       = {}
    level.sweeps      = 1

    if getattr(feval, 'forcing', None) is not None:
      level.forcing = True

    self.levels.append(level)


  @property
  def nlevels(self):
    """Return the number of levels."""
    return len(self.levels)


  #############################################################################

  def add_hook(self, level, location, hook):
    """Add a hook to be called for, eg, diagnostics or output.

    :param level: PFASST level (integer)
    :param location: location in the algorithm (string)
    :param hook: callable

    Valid locations are:

    * ``'pre-predictor'``: hook called before the predictor (only
      called at the coarsest level);

    * ``'post-predictor'``: hook called after the predictor (only
      called at the coarsest level);

    * ``'pre-iteration'``: hook called at the beginning of each PFASST
      iteration (only called at the finest level);

    * ``'post-iteration'``: hook called after each PFASST iteration
      (only called at the finest level);

    * ``'end-step'``: hook called at the end of each time step (only
      called at the finest level);

    * ``'pre-sweep'``: hook called before each SDC sweep;

    * ``'post-sweep'``: hook called after each SDC sweep;

    * ``'pre-restrict'``: hook called before restricting in time and
      space;

    * ``'post-restrict'``: hook called after restricting in time and
      space;

    * ``'pre-interpolate'``: hook called before interpolating in time
      and space;

    * ``'post-interpolate'``: hook called after interpolating in time
      and space;


    The callable ``hook`` is passed two arguments:

    :param level: PFASST level
    :param state: PFASST state

    """

    if location not in self.levels[level].hooks:
      self.levels[level].hooks[location] = []

    self.levels[level].hooks[location].append(hook)


  #############################################################################

  def run(self,
          q0=None, dt=None, tend=None, iterations=12,
          RK=None, **kwargs):
    """Run the PFASST solver.

    :param q0: initial condition (flat numpy array, real or complex)
    :param dt: time step (float)
    :param tend: final time (float)
    :param iterations: number of PFASST iterations (integer)

    Any other keyward arguments are cascaded down through all
    subsequent function calls (ie, the evaluators, interpolators,
    restrictors, hooks etc).  This allows you to pass quick'n'dirty
    options to your functions.

    If only one level has been added, this uses a serial SDC algorithm
    to integrate in time.

    If more than one level has been added, this uses the PFASST
    algorithm to integrate in time.
    """

    if q0 is None:
      # auto add was used...
      q0 = self.q0

    if len(self.levels) == 0:
      raise ValueError, 'no levels have been added yet'

    if RK is not None:
      # use the IMEX Runge-Kutta (for serial runs only)
      # XXX: this an ugly way of doing this...
      runner = rk.ARKRunner(RK)
    else:
      # use the SDC integrators
      if len(self.levels) == 1:
        runner = serial.SerialRunner()
      else:
        runner = parallel.ParallelRunner()

    runner.mpi    = self.mpi
    runner.state  = self.state
    runner.levels = self.levels

    self.state.reset()
    self.state.nlevels = self.nlevels

    # run!
    T = self.levels[0]
    T.call_hooks('pre-total')

    runner.allocate(q0.dtype)
    runner.sanity_checks()

    T.call_hooks('pre-run')
    runner.run(q0, dt, tend, iterations, **kwargs)

    self.state.reset()
    T.call_hooks('post-run')
    T.call_hooks('post-total')
