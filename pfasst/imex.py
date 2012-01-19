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

  The explicit piece is *f1*.  The implicit piece is *f2*.

  """

  def __init__(self):

    self.pieces = 2


  def evaluate(self, y, t, f, **kwargs):
    r"""Evaluate function values *f(y, t)*.

    :param y: y (numpy array)
    :param t: time (float)
    :param f: result (numpy array)

    The result is stored in *f*.
    """

    self.f1_evaluate(y, t, f[0], **kwargs)
    self.f2_evaluate(y, t, f[1], **kwargs)


  def f1_evaluate(self, y, t, f1, **kwargs):
    r"""Evaluate explicit function *f1(y, t)*.

    :param y: y (numpy array)
    :param t: time (float)
    :param f1: result (numpy array)

    The result is stored in *f1*.

    By default, this sets *f1* to 0.
    """

    f1[...] = 0.0

  def f2_evaluate(self, y, t, f2, **kwargs):
    r"""Evaluate implicit function *f2(y, t)*.

    :param y: y (numpy array)
    :param t: time (float)
    :param f2: result (numpy array)

    The result is stored in *f2*.

    By default, this is sets *f2* to 0.
    """

    f2[...] = 0.0


  def f2_solve(self, rhs, y, t, dt, f2, **kwargs):
    r"""Solve implicit equation *f2(y,t+dt) = rhs* and evaluate
    implicit function *f2(y,t)*.

    :param rhs: right hand side (numpy array)
    :param y: y result (numpy array)
    :param t: time (float)
    :param dt: time step (float)
    :param f2: result (numpy array)

    The solution is stored in *y*, and the result is stored in *f2*.

    By default, this sets *y* to *rhs* and *f2* to 0.
    """

    y[...] = rhs
    f2[...] = 0.0


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


  def sweep(self, b, t0, dt, qSDC, fSDC, feval, gSDC=None, **kwargs):
    r"""Perform one SDC sweep.

    :param b:    right hand side (numpy array of size ``(nnodes,nqvar)``)
    :param t0:   initial time
    :param dt:   time step
    :param qSDC: solution (numpy array of size ``(nnodes,nqvar)``)
    :param fSDC: function (numpy array of size ``(nnodes,nfvar)``)
    :param feval: implicit/explicit function evaluator (instance
            of :py:class:`pfasst.feval.FEval`)

    Note that *qSDC* and *fSDC* are over-written.

    The sweep performed uses forward/backward Euler time-stepping:

    .. math::

      \begin{multline}
        q^{k+1}_{m+1} = q^{k+1}_m + \Delta t_m
            \bigl[ f_I(t_{m+1}, q^{k+1}_{m+1}) +
                 f_E(t_{m}, q^{k+1}_{m}) \bigr] \\
        + S^{m,m+1}_E \, f_E(\vec{t}, \vec{q}^{k})
        + S^{m,m+1}_I \, f_I(\vec{t}, \vec{q}^{k}) + b_{m+1}
      \end{multline}

    where :math:`m = 0 \ldots M`.  Note that the initial condition
    :math:`q^{k+1}_0` is assumed to be stored in ``b[0]``.

    """

    exp = self.smat_exp
    imp = self.smat_imp

    pieces = fSDC.shape[0]
    nnodes = fSDC.shape[1]
    shape  = fSDC.shape[2:]
    size   = feval.size

    # flatten so we can use np.dot
    fSDCf = fSDC.reshape((pieces, nnodes, size))

    # integrate f
    rhs = dt * (np.dot(exp, fSDCf[0]) + np.dot(imp, fSDCf[1]))
    rhs = rhs.reshape((nnodes-1,)+shape)

    # add b
    if b is not None:
      rhs += b[1:]

    # set initial condition and eval
    qSDC[0] = b[0]

    self.level.state.node = 0
    self.level.call_hooks('pre-feval', **kwargs)

    feval.f1_evaluate(qSDC[0], t0, fSDC[0,0], **kwargs)
    feval.f2_evaluate(qSDC[0], t0, fSDC[1,0], **kwargs)

    self.level.call_hooks('post-feval', **kwargs)

    if gSDC is not None:
      fSDC[0,0] += gSDC[0]

    # sub time-stepping
    t = t0
    dtsdc = dt * self.dsdc

    for m in range(self.nnodes-1):
      t += dtsdc[m]

      y = qSDC[m] + dtsdc[m]*fSDC[0,m] + rhs[m]

      self.level.state.node = m+1
      self.level.call_hooks('pre-feval', **kwargs)

      feval.f2_solve(y, qSDC[m+1], t, dtsdc[m], fSDC[1,m+1], **kwargs)
      feval.f1_evaluate(qSDC[m+1], t, fSDC[0,m+1], **kwargs)

      self.level.call_hooks('post-feval', **kwargs)

      if gSDC is not None:
        fSDC[0,m+1] += gSDC[m+1]

    self.level.state.node = -1
