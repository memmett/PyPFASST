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


def _restrict_time_space(qSDCF, qSDCG, F, G, **kwargs):

  nnodesF = qSDCF.shape[0]
  shapeF  = qSDCF.shape[1:]

  nnodesG = qSDCG.shape[0]
  shapeG  = qSDCG.shape[1:]

  # restrict required fine nodes
  qSDCFr = {}
  for m in range(nnodesF):
    if any(F.rmask[:, m]):
      qSDCFr[m] = np.zeros(shapeG, dtype=G.qSDC.dtype)
      F.restrict(qSDCF[m], qSDCFr[m],
                 fevalF=F.feval, fevalG=G.feval, **kwargs)

  # apply restriction matrix
  for i in range(nnodesG):
    qSDCG[i] = 0.0

    for j in range(nnodesF):
      if F.rmask[i, j]:
        qSDCG[i] += F.rmat[i, j] * qSDCFr[j]



def restrict_time_space(t0, dt, F, G, restrict_functions=False, **kwargs):
  """Restrict *F* in both time and space."""

  nnodesF = F.qSDC.shape[0]
  nnodesG = G.qSDC.shape[0]

  tratio = (nnodesF - 1) / (nnodesG - 1)

  G.call_hooks('pre-restrict', **kwargs)

  # restrict qSDC
  _restrict_time_space(F.qSDC, G.qSDC, F, G, **kwargs)

  # restrict tau
  if F.tau is not None:

    # convert fine tau from 'node to node' to '0 to node'
    tauF = np.zeros((nnodesF,) + F.tau.shape[1:], dtype=F.tau.dtype)
    for m in range(1, nnodesF):
      tauF[m] = tauF[m-1] + F.tau[m-1]

    # restrict
    tauG = np.zeros((nnodesG,) + G.tau.shape[1:], dtype=G.tau.dtype)
    _restrict_time_space(tauF, tauG, F, G, **kwargs)

    # convert coarse tau from '0 to node' to 'node to node'
    for m in range(nnodesG-1, 1, -1):
      tauG[m] -= tauG[m-1]

    # set coarse tau
    G.tau[...] = tauG[1:]

  else:
    G.tau[...] = 0.0


  # re-evaluate
  if restrict_functions:
    for p in range(F.feval.pieces):
      _restrict_time_space(F.fSDC[p], G.fSDC[p], F, G, **kwargs)
  else:
    G.sdc.evaluate(t0, G.qSDC, G.fSDC, 'all', G.feval, **kwargs)


  # fas
  G.tau += fas(dt, F.fSDC, G.fSDC, F, G, **kwargs)

  G.call_hooks('post-restrict', **kwargs)
