"""PyPFASST ParallelRunner class."""

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


import math
import numpy as np

from restrict import restrict_time_space
from restrict import restrict_space_sum_time
from interpolate import interpolate_correction_time_space as interpolate_time_space
from interpolate import interpolate_correction as interpolate
from interpolate import time_interpolation_matrix
from fas import fas

from runner import Runner

from options import db as optdb


def eval_at_sdc_nodes(t0, dt, qSDC, fSDC, F, **kwargs):
  """Evaluate implicit/explicit functions at each SDC substep."""

  for m in range(len(F.sdc.nodes)):
    t = t0 + dt*F.sdc.nodes[m]

    F.feval.evaluate(qSDC[m], t, fSDC[:,m], **kwargs)

  if F.forcing:
    fSDC[0] += F.gSDC


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
      levels[l].qsend = np.zeros(shape, dtype=dtype) # send buffer
      levels[l].qrecv = np.zeros(shape, dtype=dtype) # recv buffer
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

  def cycles(self, cycle):
    """Build cycles."""

    nlevels = len(self.levels)

    cycles = []

    if cycle == 'V':
      down = []
      up   = []

      for l in range(nlevels):
        down.append(l)

      for l in range(nlevels-2, -1, -1):
        up.append(l)

      cycles.append((down, up))

    else:
      raise ValueError("cycle '%s' not understood" % cycle)

    return cycles


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
    T.feval.evaluate(T.qSDC[0], t0, T.fSDC[:,0])
    for n in range(1, T.sdc.nnodes):
      T.qSDC[n]   = T.qSDC[0]
      T.fSDC[:,n] = T.fSDC[:,0]

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

      eval_at_sdc_nodes(t0, dt, G.qSDC, G.fSDC, G, **kwargs)

      G.tau += fas(dt, F.fSDC, G.fSDC, F, G, **kwargs)

    for F in self.levels:
      F.q0[...] = F.qSDC[0]


  #############################################################################

  def predictor(self, t0, dt, **kwargs):
    """Perform the PFASST predictor."""

    B = self.levels[-1]
    rank  = self.mpi.rank
    ntime = self.mpi.ntime

    self.state.cycle = -1
    self.state.iteration = -1
    self.state.predictor = True
    B.call_hooks('pre-predictor', **kwargs)

    pred_iters = optdb.predictor_iterations or 1
    for k in range(1, rank+pred_iters+1):
      self.state.iteration = k

      # get new initial value (skip on first iteration)
      if k > 1 and rank > 0:
        B.receive(k-1, blocking=True, **kwargs)

      # coarse sdc sweep
      B.call_hooks('pre-predictor-sweep', **kwargs)

      for s in range(B.sweeps):
        B.sdc.sweep(B.q0, t0, dt, B.qSDC, B.fSDC, B.feval,
                    tau=B.tau, gSDC=B.gSDC, **kwargs)
      B.qend[...] = B.qSDC[-1]

      B.call_hooks('post-predictor-sweep', **kwargs)

      # send result forward
      if rank < ntime-1:
        B.send(k, blocking=True)

    self.state.iteration = -1
    B.call_hooks('post-predictor', **kwargs)
    self.state.predictor = False

    # interpolate coarest to finest and set initial conditions
    for F, G in self.coarse_to_fine:
      interpolate_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
      eval_at_sdc_nodes(t0, dt, F.qSDC, F.fSDC, F, **kwargs)

    for F in self.levels:
      F.q0[...] = F.qSDC[0]

    # XXX: sweep in middle levels?

    for F in self.levels:
      F.call_hooks('end-predictor', **kwargs)



  #############################################################################

  def iteration(self, k, t0, dt, cycles, **kwargs):
    """Perform one PFASST iteration."""

    levels  = self.levels
    nlevels = len(levels)

    rank  = self.mpi.rank
    ntime = self.mpi.ntime

    T = levels[0]               # finest/top level
    B = levels[nlevels-1]       # coarsest/bottom level

    self.state.cycle     = 0
    T.call_hooks('pre-iteration', **kwargs)

    # post receive requests
    if rank > 0:
      for iF in range(len(self.levels)-1):
        F = self.levels[iF]
        F.post_receive((F.level+1)*100+k)

    #### cycle

    for down, up in cycles:

      #### down

      for iF in down:
        self.state.cycle += 1

        finest   = iF == 0
        coarsest = iF == nlevels - 1

        F = levels[iF]
        if not coarsest:
          G = levels[iF+1]

        # get new initial value on coarsest level

        if coarsest:
          if rank > 0:
            F.receive((F.level+1)*100+k, blocking=coarsest, **kwargs)

        # sdc sweep

        F.call_hooks('pre-sweep', **kwargs)

        for s in range(F.sweeps):
          F.sdc.sweep(F.q0, t0, dt, F.qSDC, F.fSDC, F.feval,
                      tau=F.tau, gSDC=F.gSDC, **kwargs)
        F.qend[...] = F.qSDC[-1]

        F.call_hooks('post-sweep', **kwargs)

        # send new value forward

        if rank < ntime-1:
          F.send((F.level+1)*100+k, blocking=coarsest)

        # restrict

        if not coarsest:
          G.call_hooks('pre-restrict', **kwargs)

          restrict_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
          restrict_space_sum_time(F.tau, G.tau, F, G, **kwargs)
          eval_at_sdc_nodes(t0, dt, G.qSDC, G.fSDC, G, **kwargs)

          G.tau += fas(dt, F.fSDC, G.fSDC, F, G, **kwargs)

          G.call_hooks('post-restrict', **kwargs)

      #### up

      for iF in up:
        self.state.cycle += 1

        finest = iF == 0

        F = levels[iF]
        G = levels[iF+1]

        # interpolate

        G.call_hooks('pre-interpolate', **kwargs)

        interpolate_time_space(F.qSDC, G.qSDC, F, G, **kwargs)
        eval_at_sdc_nodes(t0, dt, F.qSDC, F.fSDC, F, **kwargs)

        G.call_hooks('post-interpolate', **kwargs)

        # get new initial value

        if rank > 0:
          F.receive((F.level+1)*100+k, **kwargs)
          interpolate(F.q0, G.q0, F, G, **kwargs)

        # sdc sweep

        if not finest:

          F.call_hooks('pre-sweep', **kwargs)

          for s in range(F.sweeps):
            F.sdc.sweep(F.q0, t0, dt, F.qSDC, F.fSDC, F.feval,
                        tau=F.tau, gSDC=F.gSDC, **kwargs)
          F.qend[...] = F.qSDC[-1]

          F.call_hooks('post-sweep', **kwargs)

    #### done

    self.state.cycle = 0
    T.call_hooks('post-iteration', **kwargs)


  #############################################################################

  def run(self, q0=None, dt=None, tend=None, iterations=None, cycle='V', **kwargs):
    """Run in parallel (PFASST)."""

    if q0 is None:
      raise ValueError, 'missing initial condition'

    #### short cuts, state, options

    levels  = self.levels
    nlevels = len(levels)

    rank  = self.mpi.rank
    ntime = self.mpi.ntime

    T = levels[0]               # finest/top level
    B = levels[nlevels-1]       # coarsest/bottom level

    self.state.dt   = dt
    self.state.tend = tend

    iterations = optdb.iterations or iterations
    cycles = self.cycles(optdb.cycle or cycle)

    #### build time interpolation matrices
    for l in range(nlevels-1):
      nodesF = levels[l].sdc.nodes
      nodesG = levels[l+1].sdc.nodes

      levels[l].time_interp_mat = time_interpolation_matrix(nodesF, nodesG)


    #### block loop
    nblocks = int(math.ceil(tend/(dt*ntime)))

    for block in range(nblocks):

      step = block*ntime + rank
      t0   = dt*step

      self.state.t0     = t0
      self.state.block  = block
      self.state.step   = step

      self.set_initial_conditions(q0, t0, dt, **kwargs)
      self.predictor(t0, dt, **kwargs)

      # iterations
      for k in range(iterations):
        self.state.iteration = k
        self.iteration(k, t0, dt, cycles, **kwargs)

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
