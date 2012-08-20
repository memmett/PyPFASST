"""PyPFASST ParallelRunner class."""

# Copyright (c) 2011, 2012 Matthew Emmett.  All rights reserved.
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


import math
import numpy as np

from restrict import restrict_time_space
from restrict import restrict_space_sum_time
from interpolate import interpolate_correction_time_space as interpolate_time_space
from interpolate import interpolate_correction as interpolate
from interpolate import time_interpolation_matrix
from fas import fas

from runner import Runner


def identity_interpolator(yF, yG, **kwargs):
  """Identity interpolator (simply copies)."""
  yF[...] = yG[...]


def identity_restrictor(yF, yG, **kwargs):
  """Identity interpolator (simply copies)."""
  yG[...] = yF[...]


###############################################################################

class ParallelRunner(Runner):
  """Parallel PFASST class/driver."""


  #############################################################################

  def allocate(self, dtype):

    levels  = self.levels
    nlevels = len(levels)

    for l in range(nlevels):
      shape  = levels[l].feval.shape
      pieces = levels[l].feval.pieces
      nnodes = levels[l].sdc.nnodes

      levels[l].q0    = np.zeros(shape, dtype=dtype)
      levels[l].qend  = np.zeros(shape, dtype=dtype)
      levels[l].qsend = np.zeros(shape, dtype=dtype)
      levels[l].qrecv = np.zeros(shape, dtype=dtype)
      levels[l].qSDC  = np.zeros((nnodes,)+shape, dtype=dtype)
      levels[l].fSDC  = np.zeros((pieces,nnodes,)+shape, dtype=dtype)

      if l > 0:
        levels[l].tau = np.zeros((nnodes-1,)+shape, dtype=dtype)

      if levels[l].forcing:
        levels[l].gSDC = np.zeros((nnodes,)+shape, dtype=dtype)

    self.fine_to_coarse = []
    for l in range(nlevels-1):
      self.fine_to_coarse.append((levels[l], levels[l+1]))

    self.coarse_to_fine = []
    for l in range(nlevels-1, 0, -1):
      self.coarse_to_fine.append((levels[l-1], levels[l]))


  #############################################################################

  def sanity_checks(self):

    # check that sdc nodes between levels overlap
    for F, G in self.fine_to_coarse:
      tratio = (F.sdc.nnodes - 1) / (G.sdc.nnodes - 1)

      d = abs(F.sdc.nodes[::tratio] - G.sdc.nodes).max()
      if d > 1e-12:
        raise ValueError, "SDC nodes don't overlap"

    # check first and last SDC nodes
    for F in self.levels:
      if F.sdc.nodes[0] != 0.0:
        raise ValueError, "first SDC node should be 0.0"

      if F.sdc.nodes[-1] != 1.0:
        raise ValueError, "last SDC node should be 1.0"

    # set "identity" interpolator and restrictor if necessary
    for F, G in self.fine_to_coarse:
      if F.interpolate is None:
        F.interpolate = identity_interpolator

      if F.restrict is None:
        F.restrict = identity_restrictor


  #############################################################################

  def set_initial_conditions(self, q0, t0, dt, **kwargs):
    """Set initial conditions."""

    levels = self.levels
    T = levels[0]

    try:
      T.q0[...] = q0
    except ValueError:
      raise ValueError, 'initial condition shape mismatch'

    # set initial condtion
    T.qSDC[0] = T.q0

    # evaluate at first node and spread
    T.sdc.evaluate(t0, T.qSDC, T.fSDC, 0, T.feval, **kwargs)
    for n in range(1, T.sdc.nnodes):
      T.qSDC[n]   = T.qSDC[0]
      for p in range(T.fSDC.shape[0]):
        T.fSDC[p,n] = T.fSDC[p,0]


    # evaluate forcing terms
    for F in self.levels:
      if F.forcing:
        for m in range(len(F.sdc.nodes)):
          t = t0 + dt*F.sdc.nodes[m]
          F.feval.forcing(t, F.gSDC[m], **kwargs)

    if T.forcing:
      T.fSDC[0] += T.gSDC

    # restrict finest level to coarser levels
    for F, G in self.fine_to_coarse:
      restrict_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
      restrict_space_sum_time(F.tau, G.tau, F, G, **kwargs)

      G.sdc.evaluate(t0, G.qSDC, G.fSDC, 'all', G.feval, **kwargs)

      G.tau += fas(dt, F.fSDC, G.fSDC, F, G, **kwargs)

    for F in self.levels:
      F.q0[...] = F.qSDC[0]


  #############################################################################

  def predictor(self, t0, dt, **kwargs):
    """Perform the PFASST predictor."""

    B     = self.levels[-1]
    rank  = self.mpi.rank
    ntime = self.mpi.ntime

    self.state.cycle = -1
    self.state.iteration = -1
    self.state.predictor = True
    B.call_hooks('pre-predictor', **kwargs)

    for k in range(1, rank+2):
      self.state.iteration = k

      if k > 1:
        B.receive(k-1, blocking=True)

      for s in range(B.sweeps):
        B.sdc.sweep(t0, dt, B, **kwargs)

      B.send(k, blocking=True)

    self.state.iteration = -1
    B.call_hooks('post-predictor', **kwargs)
    self.state.predictor = False

    # interpolate coarest to finest and set initial conditions
    for F, G in self.coarse_to_fine:
      # XXX: sweep in middle levels?
      interpolate_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
      F.sdc.evaluate(t0, F.qSDC, F.fSDC, 'all', F.feval, **kwargs)

    for F in self.levels:
      F.q0[...] = F.qSDC[0]

    for F in self.levels:
      F.call_hooks('end-predictor', **kwargs)


  #############################################################################

  def iteration(self, t0, dt, **kwargs):
    """Perform one PFASST iteration."""

    levels  = self.levels
    nlevels = len(levels)

    rank  = self.mpi.rank
    ntime = self.mpi.ntime
    k     = self.state.iteration

    T = levels[0]               # finest/top level
    B = levels[nlevels-1]       # coarsest/bottom level

    self.state.cycle = 0
    T.call_hooks('pre-iteration', **kwargs)


    #### post receive requests

    for F in levels[:-1]:
      F.post_receive((F.level+1)*100+k)


    #### down

    for F, G in self.fine_to_coarse:
      self.state.cycle += 1

      for s in range(F.sweeps):
        F.sdc.sweep(t0, dt, F, **kwargs)

      F.send((F.level+1)*100+k, blocking=False)

      G.call_hooks('pre-restrict', **kwargs)

      restrict_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
      restrict_space_sum_time(F.tau, G.tau, F, G, **kwargs)
      G.sdc.evaluate(t0, G.qSDC, G.fSDC, 'all', G.feval, **kwargs)

      G.tau += fas(dt, F.fSDC, G.fSDC, F, G, **kwargs)

      G.call_hooks('post-restrict', **kwargs)


    #### bottom

    B.receive((B.level+1)*100+k, blocking=True)

    for s in range(B.sweeps):
      B.sdc.sweep(t0, dt, B, **kwargs)

    B.send((B.level+1)*100+k, blocking=True)


    #### up

    for F, G in self.coarse_to_fine:
      self.state.cycle += 1

      # interpolate

      G.call_hooks('pre-interpolate', **kwargs)

      interpolate_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
      F.sdc.evaluate(t0, F.qSDC, F.fSDC, 'all', F.feval, **kwargs)

      G.call_hooks('post-interpolate', **kwargs)

      # get new initial value

      F.receive((F.level+1)*100+k, **kwargs)
      if rank > 0:
        interpolate(F.q0, G.q0, F, G, **kwargs)

      # sweep

      if F.level != 0:
        for s in range(F.sweeps):
          F.sdc.sweep(t0, dt, F, **kwargs)


    #### done

    self.state.cycle = 0
    T.call_hooks('post-iteration', **kwargs)


  #############################################################################

  def run(self, q0=None, dt=None, tend=None, iterations=None, **kwargs):
    """Run in parallel (PFASST)."""

    if q0 is None:
      raise ValueError, 'missing initial condition'


    #### short cuts, state

    levels  = self.levels

    rank  = self.mpi.rank
    ntime = self.mpi.ntime

    self.state.dt   = dt
    self.state.tend = tend


    #### build time interpolation matrices

    for F, G in self.fine_to_coarse:
      F.time_interp_mat = time_interpolation_matrix(F.sdc.nodes, G.sdc.nodes)


    #### block loop

    nblocks = int(math.ceil(tend/(dt*ntime)))

    for block in range(nblocks):

      step = block*ntime + rank
      t0   = dt*step

      self.state.t0    = t0
      self.state.block = block
      self.state.step  = step

      self.set_initial_conditions(q0, t0, dt, **kwargs)
      self.predictor(t0, dt, **kwargs)

      # iterations
      for k in range(iterations):
        self.state.iteration = k
        self.iteration(t0, dt, **kwargs)

      # loop
      if nblocks > 1 and block < nblocks:
        if self.mpi.nspace > 1:
          raise ValueError, 'looping not implemented for nspace > 1'

        self.mpi.comm.Bcast(T.qend, root=self.mpi.ntime-1)
        q0 = T.qend

      for F in levels:
        F.call_hooks('end-step', **kwargs)


    #### done

    for F in levels:
      F.call_hooks('end', **kwargs)
