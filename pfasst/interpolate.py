"""PyPFASST interpolation routines."""

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


def interpolate_correction(qF, qG, F, G, **kwargs):
  """**Adjust** *qF* by interpolating the *qG* correction.

  The different between *qG* and *qF* (after restricting) is
  interpolated and added to *qF*.
  """

  # compute coarse increment
  qFr = np.empty(qG.shape, dtype=qG.dtype)
  F.restrict(qF, qFr, fevalF=F.feval, fevalG=G.feval, **kwargs)
  delG = qG - qFr

  # interpolate
  delF = np.empty(qF.shape, dtype=qF.dtype)
  F.interpolate(delF, delG, fevalF=F.feval, fevalG=G.feval, **kwargs)

  qF += delF


###############################################################################

def interpolate_correction_time_space(t0, F, G, **kwargs):
  """**Adjust** *qSDCF* by interpolating the *qSDCG* correction in
  both space and time."""

  nnodesF = F.sdc.nnodes
  shapeF  = F.feval.shape
  sizeF   = F.feval.size

  nnodesG = G.sdc.nnodes
  shapeG  = G.feval.shape
  sizeG   = G.feval.size

  tratio = (nnodesF - 1) / (nnodesG - 1)

  if ((sizeF == sizeG) and (nnodesF == nnodesG)):
    F.qSDC[...] = G.qSDC
    return

  G.call_hooks('pre-interpolate', **kwargs)

  #### compute coarse corrections

  delG_C = G.qSDC.copy()

  # restrict required fine nodes
  qSDCFr = {}

  for m in range(nnodesF):
    if any(F.rmask[:, m]):
      qSDCFr[m] = np.zeros(shapeG, dtype=G.qSDC.dtype)
      F.restrict(F.qSDC[m], qSDCFr[m],
                 fevalF=F.feval, fevalG=G.feval, **kwargs)

  # apply restriction matrix to compute coarse corrections
  for i in range(nnodesG):
    for j in range(nnodesF):
      if F.rmask[i, j]:
        delG_C[i] -= F.rmat[i, j] * qSDCFr[j]

  del qSDCFr


  #### interpolate increments in time

  delG_F = np.empty((nnodesG,) + shapeF, dtype=F.qSDC.dtype)

  for m in range(nnodesG):
    F.interpolate(delG_F[m], delG_C[m],
                  fevalF=F.feval, fevalG=G.feval, **kwargs)

  # apply interpolation matrix to compute fine corrections
  for i in range(nnodesF):
    for j in range(nnodesG):
      if F.tmask[i, j]:
        F.qSDC[i] += F.tmat[i, j] * delG_F[j]


  #### re-evaluate

  F.sdc.evaluate(t0, F.qSDC, F.fSDC, 'all', F.feval, **kwargs)

  G.call_hooks('post-interpolate', **kwargs)
  



###############################################################################

def time_interpolation_matrix(nodesF, nodesG, dtype=np.float64):
  """Return the polynomial interpolation matrix required to
  interpolate values to missing SDC nodes between levels."""

  nnodesF = len(nodesF)
  nnodesG = len(nodesG)

  tmat  = np.zeros((nnodesF, nnodesG), dtype=dtype)
  tmask = np.zeros((nnodesF, nnodesG), dtype=np.int8)

  for i in range(nnodesF):
    xi = nodesF[i]

    for j in range(nnodesG):

      den = 1.0
      num = 1.0

      ks = range(nnodesG)
      ks.remove(j)

      for k in ks:
        den = den * (nodesG[j] - nodesG[k])
        num = num * (xi        - nodesG[k])

      tmat[i, j] = num/den

      if abs(num) > 1e-12:
        tmask[i, j] = True

  return tmat, tmask
