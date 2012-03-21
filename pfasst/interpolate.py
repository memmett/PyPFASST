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

def interpolate_correction_time_space(qSDCF, qSDCG, F, G, **kwargs):
  """**Adjust** *qSDCF* by interpolating the *qSDCG* correction in
  both space and time."""

  nnodesF = F.sdc.nnodes
  nnodesG = G.sdc.nnodes

  sizeF = F.feval.size
  sizeG = G.feval.size

  tratio = (nnodesF - 1) / (nnodesG - 1)

  if ((sizeF == sizeG) and (nnodesF == nnodesG)):
    qSDCF[...] = qSDCG
    return

  # compute coarse increments
  qSDCFr = np.zeros(qSDCG.shape, dtype=qSDCG.dtype)

  for m in range(nnodesG):
    mf = m*tratio
    F.restrict(qSDCF[mf], qSDCFr[m],
               fevalF=F.feval, fevalG=G.feval, **kwargs)

  delG = qSDCG - qSDCFr

  # interpolate increments in time
  delGF  = np.empty(qSDCF.shape, dtype=qSDCF.dtype)
  delGFf = delGF.reshape((nnodesF,sizeF))

  for m in range(nnodesG):
    mf = m*tratio
    F.interpolate(delGF[mf], delG[m],
                  fevalF=F.feval, fevalG=G.feval, **kwargs)

  # interpolate inbetween nodes
  if tratio != 1:
    delGFf[1:nnodesF:tratio] = np.dot(F.time_interp_mat, delGFf[::tratio])


  qSDCF += delGF


###############################################################################

def time_interpolation_matrix(nodesF, nodesG, dtype=np.float64):
  """Return the polynomial interpolation matrix required to
  interpolate values to missing SDC nodes between levels."""

  nnodesF = len(nodesF)
  nnodesG = len(nodesG)

  ndiff  = nnodesF - nnodesG
  tratio = (nnodesF - 1) / (nnodesG - 1)

  if tratio == 1:
    interp_mat = 1.0

  elif tratio == 2:
    interp_mat = np.zeros((ndiff,nnodesG), dtype=dtype)

    for i in range(ndiff):
      xi = nodesF[i*tratio+1]

      for j in range(nnodesG):

        den = 1.0
        num = 1.0

        ks = range(nnodesG)
        ks.remove(j)

        for k in ks:
          den = den * (nodesG[j] - nodesG[k])
          num = num * (xi    - nodesG[k])

        interp_mat[i,j] = num/den

  else:
    raise NotImplementedError, 'time ratio must be 1 or 2 (currently)'


  return interp_mat
