"""PyPFASST serial PFASST class."""

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

from runner import Runner


class SerialRunner(Runner):
  """Serial PFASST class/driver."""


  #############################################################################

  def allocate(self, dtype):

    levels  = self.levels
    nlevels = len(levels)

    for l in range(nlevels):
      shape  = levels[l].feval.shape
      pieces = levels[l].feval.pieces
      nnodes = levels[l].sdc.nnodes

      levels[l].q0   = np.zeros(shape, dtype=dtype)
      levels[l].qend = np.zeros(shape, dtype=dtype)
      levels[l].qSDC = np.zeros((nnodes,)+shape, dtype=dtype)
      levels[l].fSDC = np.zeros((pieces,nnodes,)+shape, dtype=dtype)

      if levels[l].forcing:
        levels[l].gSDC = np.zeros((nnodes,)+shape, dtype=dtype)


  #############################################################################

  def run(self, q0=None, dt=None, tend=None, iterations=None, **kwargs):
    """Run in serial (SDC)."""

    #### short cuts, state, options

    F = self.levels[0]

    self.state.dt   = dt
    self.state.tend = tend


    #### set initial condition

    if q0 is None:
      raise ValueError, 'missing initial condition'

    try:
      F.q0[...] = q0
    except ValueError:
      raise ValueError, 'initial condition shape mismatch'


    #### time "block" loop

    nblocks = int(math.ceil(tend/dt))

    for block in range(nblocks):

      t0 = dt*block

      self.state.t0     = t0
      self.state.block  = block
      self.state.step   = block

      F.call_hooks('pre-iteration', **kwargs)

      # set initial condtion
      F.qSDC[0] = F.q0

      # evaluate at first node and spread
      F.feval.evaluate(F.qSDC, t0, F.fSDC, 0, **kwargs)
      for n in range(1, F.sdc.nnodes):
        F.qSDC[n]   = F.qSDC[0]
        for p in range(F.fSDC.shape[0]):
          F.fSDC[p,n] = F.fSDC[p,0]

      if F.forcing:
        for m in range(len(F.sdc.nodes)):
          t = t0 + dt*F.sdc.nodes[m]
          F.feval.forcing(t, F.gSDC[m], **kwargs)

        F.fSDC[0] += F.gSDC

      # sdc sweeps
      for k in range(iterations):
        self.state.iteration = k

        F.call_hooks('pre-sweep', **kwargs)

        for s in range(F.sweeps):
          F.sdc.sweep(F.q0, t0, dt, F.qSDC, F.fSDC, F.feval,
                      gSDC=F.gSDC, **kwargs)
        F.qend[...] = F.qSDC[-1]

        F.call_hooks('post-sweep', **kwargs)

        # XXX: check residual and break if appropriate

      F.call_hooks('post-iteration', **kwargs)

      F.call_hooks('end-step', **kwargs)

      # set next initial condition
      F.q0[...] = F.qend[...]

    F.call_hooks('end', **kwargs)
