"""PyPFASST FAS routines."""

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


import numpy as np

from options import db as optdb


def fas(dt, fSDCF, fSDCG, F, G, **kwargs):
  """Return FAS correction between *fSDCF* and *fSDCG*."""

  if not optdb.use_fas:
    return np.zeros((G.sdc.nnodes-1,)+G.feval.shape, dtype=fSDCF.dtype)

  # note: even if the number of variables and nodes are the same, we
  # should still compute the FAS correction since the function
  # evaluations may be different (eg, lower order operator for G)

  sizeF = F.feval.size
  sizeG = G.feval.size

  fF = np.empty((F.sdc.nnodes,)+F.feval.shape, dtype=fSDCF.dtype)
  fG = np.empty((G.sdc.nnodes,)+G.feval.shape, dtype=fSDCG.dtype)

  fFf = fF.reshape((F.sdc.nnodes,sizeF))
  fGf = fG.reshape((G.sdc.nnodes,sizeG))

  for m in range(F.sdc.nnodes):
    np.sum(fSDCF[:,m], axis=0, out=fF[m])

  for m in range(G.sdc.nnodes):
    np.sum(fSDCG[:,m], axis=0, out=fG[m])

  # fine integral of fine function
  FofFf = dt * np.dot(F.sdc.smat, fFf)
  FofF  = FofFf.reshape((F.sdc.nnodes-1,)+F.feval.shape)

  # coarse integral of coarse function
  CofCf = dt * np.dot(G.sdc.smat, fGf)
  CofC  = CofCf.reshape((G.sdc.nnodes-1,)+G.feval.shape)

  # coarse value of fine integral (restrict fine to coarse, sum fine
  # nodes between coarse nodes)
  CofF = np.zeros((G.sdc.nnodes-1,)+G.feval.shape, dtype=fSDCF.dtype)
  tratio = (F.sdc.nnodes-1) / (G.sdc.nnodes-1)

  tmp = np.zeros(G.feval.shape, dtype=fSDCF.dtype)

  for m in range(F.sdc.nnodes-1):
    mc = m/tratio

    F.restrict(FofF[m], tmp, fevalF=F.feval, fevalG=G.feval, **kwargs)

    CofF[mc] = CofF[mc] + tmp

  return CofF - CofC

