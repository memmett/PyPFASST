"""PyPFASST SDC class."""

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


import numpy as np
import quadrature

class SDC(object):
  r"""SDC base class.

  :param qtype: quadrature type
  :param nodes: number of quadrature nodes
  :param refine: refinement factor

  The SDC class performs SDC sweeps and adds FAS corrections if
  supplied.  See also :py:class:`pfasst.imex.IMEXSDC`.

  Currently supported quadrature types are:

  * ``'GL'``: Gauss-Labotto
  * ``'GR'``: Gauss-Radau
  * ``'CC'``: Clenshaw-Curtis

  By default, the constructor loads the quadrature nodes and spectral
  integration matrix correcponding to *nodes* and *qtype* into the
  instance variables *nodes* and *smat*.

  """

  def __init__(self, qtype, nnodes, refine=1, dtype=np.float64):

    nodes = quadrature.table[qtype,nnodes,refine]['nodes']
    smat  = quadrature.table[qtype,nnodes,refine]['matrix']

    self.nnodes = (nnodes-1)/refine + 1
    self.type   = qtype

    self.nodes = np.array(nodes, dtype=dtype)
    self.smat  = np.array(smat,  dtype=dtype)



  #############################################################################

  def residual(self, q0, qend, dt, fSDC, feval, **kwargs):
    """Return the residual of *fSDC*."""

    f  = np.empty((self.nnodes,) + feval.shape, dtype=fSDC.dtype)
    ff = f.reshape((self.nnodes,feval.size))

    for m in range(self.nnodes):
      f[m] = fSDC[0,m]
      for p in range(1, fSDC.shape[0]):
        f[m] += fSDC[p,m]

    int_ff = dt * np.dot(self.smat, ff)
    int_f  = int_ff.reshape((self.nnodes-1,)+feval.shape)

    tot_f = np.zeros(feval.shape, dtype=fSDC.dtype)
    for m in range(self.nnodes-1):
      tot_f += int_f[m,:]

    return q0 + tot_f - qend


  #############################################################################

  def sweep(self, *args, **kwargs):
    """Perform one SDC sweep.

    **This method should be overridden.**
    """

    raise NotImplementedError()


  #############################################################################

  def evaluate(self, *args, **kwargs):
    """Evaluate.

    **This method should be overridden.**
    """

    raise NotImplementedError()
