r"""PyPFASST IMEX SDC and FEval classes.  These classes are used for
ODEs of the form

.. math::

  \frac{d}{dt} y(x,t) = f(y,t) = f_1(y,t) + f_2(y,t) + b(t)

where the solver treats the :math:`f_1` piece explicitly, and the
:math:`f_2` piece implicitly.

"""

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

import feval
import sdc


class IMEXFEval(feval.FEval):
  """IMEX function evaluation base class.

  Methods in this class should be overridden with problem specific
  evaluators.

  Attributes:

  .. attribute:: f1_evaluate

     Evaluate explicit piece.  Called as

     >>> f1_evaluate(q, t, f1, **kwargs)

     The result should be stored in *f1*.

  .. attribute:: f2_evaluate

     Evalute implicit piece.  Called as

     >>> f2_evaluate(q, t, f2, **kwargs)

     The result should be stored in *f2*.

  .. attribute:: f2_solve

     Solve and evaluate implicit piece.  Called as

     >>> f2_solve(rhs, q, t, dt, f2, **kwargs)

     The implicit solution of :math:`q - \Delta t f_2(q) =
     \text{RHS}`` should be stored in *q*.  The value of
     :math:`f_2(q)` should be stored in *f2*.

  Note that by omitting *f1eval* or both of *f2eval* and *f2solv*,
  this becomes a purely implicit or explicit evaluator, respectively.

  See also :py:class:`pfasst.feval.FEval`.


  """

  @property
  def pieces(self):

    f1eval = hasattr(self, 'f1_evaluate')
    f2eval = hasattr(self, 'f2_evaluate')
    f2solv = hasattr(self, 'f2_solve')

    if not (f1eval or f2eval or f2solv):
      raise ValueError('none of f1eval, f2eval, or f2solv are defined')

    if f1eval and f2eval and f2solv:
      return 2
    elif f1eval and not (f2eval or f2solv):
      return 1
    elif not f1eval and (f2eval and f2solv):
      return 1
    else:
      raise ValueError('feval is inconsistent')


class IMEXSDC(sdc.SDC):
  r"""IMEXSDC (implicit/explicit SDC) class.

  :param qtype: quadrature type
  :param nodes: number of quadrature nodes

  The SDC class performs implicit/explicit SDC sweeps and adds FAS
  corrections if supplied.

  Currently supported quadrature types are:

  * ``'GL'``: Gauss-Labotto
  * ``'CC'``: Clenshaw-Curtis

  """

  def __init__(self, *args, **kwargs):

    super(IMEXSDC, self).__init__(*args, **kwargs)

    # construct implicit and explicit integration matrices
    smat_exp = self.smat.copy()
    smat_imp = self.smat.copy()

    dsdc = self.nodes[1:] - self.nodes[:-1]
    for m in range(self.nnodes-1):
      smat_exp[m,m]   = smat_exp[m,m]   - dsdc[m]
      smat_imp[m,m+1] = smat_imp[m,m+1] - dsdc[m]

    self.smat_exp = smat_exp
    self.smat_imp = smat_imp

    self.dsdc = self.nodes[1:] - self.nodes[:-1]


  ###############################################################################

  def evaluate(self, t0, qSDC, fSDC, node, feval, **kwargs):

    nnodes = fSDC.shape[1]

    f1eval = hasattr(feval, 'f1_evaluate')
    f2eval = hasattr(feval, 'f2_evaluate')

    if node == 'all':

      for m in range(nnodes):

        self.pf.state.node = m
        self.level.call_hooks('pre-feval', **kwargs)

        if f1eval and f2eval:
          feval.f1_evaluate(qSDC[m], t0, fSDC[0, m], **kwargs)
          feval.f2_evaluate(qSDC[m], t0, fSDC[1, m], **kwargs)
        elif f1eval:
          feval.f1_evaluate(qSDC[m], t0, fSDC[0, m], **kwargs)
        else:
          feval.f2_evaluate(qSDC[m], t0, fSDC[0, m], **kwargs)

        self.level.call_hooks('post-feval', **kwargs)

    else:

      self.pf.state.node = node
      self.level.call_hooks('pre-feval', **kwargs)

      if f1eval and f2eval:
        feval.f1_evaluate(qSDC[node], t0, fSDC[0, node], **kwargs)
        feval.f2_evaluate(qSDC[node], t0, fSDC[1, node], **kwargs)
      elif f1eval:
        feval.f1_evaluate(qSDC[node], t0, fSDC[0, node], **kwargs)
      else:
        feval.f2_evaluate(qSDC[node], t0, fSDC[0, node], **kwargs)

      self.level.call_hooks('post-feval', **kwargs)


  ###############################################################################

  def sweep(self, t0, dt, F, **kwargs):
    r"""Perform one SDC sweep.

    Note that *qSDC* and *fSDC* are over-written.

    The sweep performed uses forward/backward Euler time-stepping.
    """

    exp = self.smat_exp
    imp = self.smat_imp

    qSDC = F.qSDC
    fSDC = F.fSDC
    feval = F.feval
    
    pieces = fSDC.shape[0]
    nnodes = fSDC.shape[1]
    shape  = fSDC.shape[2:]
    size   = feval.size

    f1eval = hasattr(feval, 'f1_evaluate')
    f2eval = hasattr(feval, 'f2_evaluate')

    F.call_hooks('pre-sweep', **kwargs)

    # flatten so we can use np.dot
    fSDCf = fSDC.reshape((pieces, nnodes, size))

    # integrate f
    if f1eval and f2eval:
      rhs = dt * (np.dot(exp, fSDCf[0]) + np.dot(imp, fSDCf[1]))
    elif f1eval:
      rhs = dt * np.dot(exp, fSDCf[0])
    else:
      rhs = dt * np.dot(imp, fSDCf[0])

    rhs = rhs.reshape((nnodes-1,)+shape)

    # add tau
    if F.tau is not None:
      rhs += F.tau

    # set initial condition and eval
    qSDC[0] = F.q0
    self.evaluate(t0, qSDC, fSDC, 0, feval, **kwargs)

    if F.gSDC is not None:
      fSDC[0,0] += F.gSDC[0]

    # sub time-stepping
    t = t0
    dtsdc = dt * self.dsdc

    for m in range(self.nnodes-1):
      t += dtsdc[m]

      self.pf.state.node = m + 1
      self.level.call_hooks('pre-feval', **kwargs)

      if f1eval and f2eval:
        # imex

        q1 = qSDC[m] + dtsdc[m]*fSDC[0,m] + rhs[m]
        feval.f2_solve(q1, qSDC[m+1], t, dtsdc[m], fSDC[1,m+1], **kwargs)
        feval.f1_evaluate(qSDC[m+1], t, fSDC[0,m+1], **kwargs)

      elif f1eval:
        # explicit

        qSDC[m+1] = qSDC[m] + dtsdc[m]*fSDC[0,m] + rhs[m]
        feval.f1_evaluate(qSDC[m+1], t, fSDC[0,m+1], **kwargs)

      else:
        # implicit

        q1 = qSDC[m] + dtsdc[m]*fSDC[0,m] + rhs[m]
        feval.f2_solve(q1, qSDC[m+1], t, dtsdc[m], fSDC[0,m+1], **kwargs)

      self.level.call_hooks('post-feval', **kwargs)

      if F.gSDC is not None:
        fSDC[0,m+1] += F.gSDC[m+1]

    F.qend[...] = F.qSDC[-1]

    self.pf.state.node = -1

    F.call_hooks('post-sweep', **kwargs)

