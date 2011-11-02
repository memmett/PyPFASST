"""PyPFASST SDC class."""

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
import warnings

class SDC(object):
  r"""SDC base class.

  :param qtype: quadrature type
  :param nodes: number of quadrature nodes

  The SDC class performs SDC sweeps and adds FAS corrections if
  supplied.  See also :py:class:`pfasst.imex.IMEXSDC`.

  Currently supported quadrature types are:

  * ``'GL'``: Gauss-Labotto
  * ``'CC'``: Clenshaw-Curtis

  By default, the constructor loads the quadrature nodes and
  spectral integration matrix correcponding to *nodes* and *qtype*
  into the instance variables *nodes* and *smat*.

  """

  def __init__(self, qtype, nnodes, refine=1, dtype=np.float64):

    # try loading pre-computed sdc nodes
    try:
      import quadrature

      nodes = quadrature.table[qtype,nnodes,refine]['nodes']
      smat  = quadrature.table[qtype,nnodes,refine]['matrix']

    except:
      # try computing symbolically
      try:
        import mpsdcquad

        (nodes, left) = mpsdcquad.nodes(qtype, nnodes, refine)
        smat = mpsdcquad.smat(nodes, left)

      except:

        raise ValueError('Invalid SDC nodes.  '
                         'SymPy is not installed, '
                         'and pre-computed SDC tables for the requested '
                         'nodes were not found.')

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
      np.sum(fSDC[:,m], axis=0, out=f[m])

    int_ff = dt * np.dot(self.smat, ff)
    int_f  = int_ff.reshape((self.nnodes-1,)+feval.shape)

    tot_f = np.zeros(feval.shape, dtype=fSDC.dtype)
    for m in range(self.nnodes-1):
      tot_f += int_f[m,:]

    return q0 + tot_f - qend


  #############################################################################

  def sweep(self, b, t0, dt, qSDC, fSDC, feval, **kwargs):
    r"""Perform one SDC sweep.

    :param b:     right hand side (numpy array of size ``(nnodes, size)``)
    :param t0:    initial time
    :param dt:    time step
    :param qSDC:  solution (numpy array of size ``(nnodes, size)``)
    :param fSDC:  function (numpy array of size ``(pieces, nnodes, size)``)
    :param feval: function evaluator (instance of
                  :py:class:`pfasst.feval.FEval`)

    **This method should be overridden.**

    See also :py:class:`pfasst.imex.IMEXSDC` and
    :py:class:`pfasst.explicit.ExplicitSDC`.

    """

    raise NotImplementedError('pfasst.sdc.sweep must be implemented')
