"""PyPFASST restrict routines."""

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

from fas import fas


def restrict_time_space(t0, dt, F, G, **kwargs):
  """Restrict *F* in both time and space."""

  nnodesF = F.qSDC.shape[0]
  nnodesG = G.qSDC.shape[0]

  tratio = (nnodesF - 1) / (nnodesG - 1)

  G.call_hooks('pre-restrict', **kwargs)

  # restrict qSDC
  for m in range(nnodesG):
    mf = m*tratio
    F.restrict(F.qSDC[mf], G.qSDC[m], fevalF=F.feval, fevalG=G.feval, **kwargs)

  # restrict tau
  G.tau[...] = 0.0

  if F.tau is not None:

    tmp = np.zeros(G.feval.shape, dtype=F.tau.dtype)

    for m in range(1, nnodesF):
      mc = int(math.ceil(1.0*m/tratio))

      F.restrict(F.tau[m-1], tmp, fevalF=F.feval, fevalG=G.feval, **kwargs)
      G.tau[mc-1] += tmp

  # re-evaluate
  G.sdc.evaluate(t0, G.qSDC, G.fSDC, 'all', G.feval, **kwargs)

  # fas
  G.tau += fas(dt, F.fSDC, G.fSDC, F, G, **kwargs)

  G.call_hooks('post-restrict', **kwargs)

  


def restrict_space_sum_time(tauF, tauG, F, G, **kwargs):
  """Restrict *tauF* in space and sum fine nodes in between the coarse
  nodes in time."""

  tauG[...] = 0.0

  if tauF is None:
    return

  nnodesF = tauF.shape[0] + 1
  nnodesG = tauG.shape[0] + 1

  tratio = (nnodesF - 1) / (nnodesG - 1)

  tmp = np.zeros(G.feval.shape, dtype=tauF.dtype)

  for m in range(1, nnodesF):
    mc = int(math.ceil(1.0*m/tratio))

    F.restrict(tauF[m-1], tmp, fevalF=F.feval, fevalG=G.feval, **kwargs)

    tauG[mc-1] += tmp
