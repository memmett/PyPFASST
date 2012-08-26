"""PyPFASST FAS routines."""

# Copyright (c) 2011, 2012, Matthew Emmett.  All rights reserved.
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


import numpy as np


###############################################################################

def restrict_sdc(qSDCF, qSDCG, F, G, **kwargs):
  """Restrict quantities defined on the fine SDC nodes to coarse SDC nodes."""

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


###############################################################################

def fas(dt, fSDCF, fSDCG, F, G, **kwargs):
  """Return FAS correction between *fSDCF* and *fSDCG*."""

  # note: even if the number of variables and nodes are the same, we
  # should still compute the FAS correction since the function
  # evaluations may be different (eg, lower order operator for G)

  nnodesF = F.sdc.nnodes
  shapeF  = F.feval.shape

  nnodesG = G.sdc.nnodes
  shapeG  = G.feval.shape

  # fine '0 to node' integral of fine function
  FofF = np.zeros((nnodesF,) + shapeF, dtype=fSDCF.dtype)
  for i in range(1, nnodesF):
    FofF[i] = FofF[i-1]
    for j in range(nnodesF):
      for p in range(fSDCF.shape[0]):
        FofF[i] += dt * F.sdc.smat[i-1, j] * fSDCF[p, j]

  # coarse value of fine integral
  CofF = np.zeros((nnodesG,) + shapeG, dtype=fSDCG.dtype)
  restrict_sdc(FofF, CofF, F, G, **kwargs)

  # coarse '0 to node' integral of coarse function
  CofG = np.zeros((nnodesG,) + shapeG, dtype=fSDCG.dtype)
  for i in range(1, nnodesG):
    CofG[i] = CofG[i-1]
    for j in range(nnodesG):
      for p in range(fSDCG.shape[0]):
        CofG[i] += dt * G.sdc.smat[i-1, j] * fSDCG[p, j]

  # convert from '0 to node' to 'node to node'
  for m in range(nnodesG-1, 1, -1):
    CofF[m] -= CofF[m-1]
    CofG[m] -= CofG[m-1]

  return (CofF - CofG)[1:]

