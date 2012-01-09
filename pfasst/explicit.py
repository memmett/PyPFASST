r"""PyPFASST explicit SDC and FEval classes.  These classes are used for
ODEs of the form

.. math::

  \frac{d}{dt} y(x,t) = f(y,t) + b(t)

where the solver treats :math:`f` explicitly.

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
import quadrature


class ExplicitFEval(feval.FEval):
  """Explicit function evaluation base class."""

  def __init__(self):
    self.pieces = 1


  def evaluate(self, y, t, f, **kwargs):
    """Evaluate function values *f(y, t)*."""

    self.f1_evaluate(y, t, f[0], **kwargs)


  def f1_evaluate(self, y, t, f, **kwargs):
    """Evaluate function values *f(y, t)*.

    :param y: y (numpy array)
    :param t: time (float)
    :param f: result (numpy array)
    """

    f[...] = 0.0


class ExplicitSDC(sdc.SDC):
  """Explicit SDC class.

  This SDC class performs explicit SDC sweeps and adds FAS corrections
  if supplied.

  """

  def __init__(self, *args, **kwargs):

    super(ExplicitSDC, self).__init__(*args, **kwargs)

    # construct implicit and explicit integration matrices
    smat_exp = self.smat.copy()

    dsdc = self.nodes[1:] - self.nodes[:-1]
    for m in range(self.nnodes-1):
      smat_exp[m,m] = smat_exp[m,m] - dsdc[m]

    self.smat_exp = smat_exp


  def sweep(self, b, t0, dt, qSDC, fSDC, feval, gSDC=None, **kwargs):
    r"""Perform one SDC sweep.

    :param t0:   initial time
    :param b:    right hand side (numpy array of size ``(nnodes,...)``)
    :param dt:   time step
    :param qSDC: solution (numpy array of size ``(nnodes,...)``)
    :param fSDC: function (numpy array of size ``(nnodes,...)``)
    :param feval: implicit/explicit function evaluator (instance
            of :py:class:`pfasst.feval.FEval`)

    Note that *qSDC* and *fSDC* are over-written.

    The sweep performed uses forward Euler time-stepping:

    .. math::

      q^{k+1}_{m+1} = q^{k+1}_m
                      + \Delta t_m f(t_m, q^{k+1}_m)
                      + S_m^{m+1} \, f(\vec{t}, \vec{q}^k)
                      + b_{m+1}

    where :math:`m = 0 \ldots M`.  Note that the initial condition
    :math:`q^{k+1}_0` is assumed to be stored in ``b[0]``.

    """

    exp = self.smat_exp

    nnodes = fSDC.shape[1]
    shape  = fSDC.shape[2:]
    size   = feval.size

    # flatten so we can use np.dot
    fSDCf = fSDC.reshape((nnodes, size))

    # integrate f
    rhs = dt * np.dot(exp, fSDCf)
    rhs = rhs.reshape((nnodes-1,)+shape)

    # add b
    if b is not None:
      rhs = rhs + b[1:]

    # set initial condition and eval
    qSDC[0] = b[0]
    feval.evaluate(qSDC[0], t0, fSDC[:,0], **kwargs)

    if gSDC is not None:
      fSDC[0,0] += gSDC[0]

    # sub time-stepping
    t = t0
    dtsdc = dt * (self.nodes[1:] - self.nodes[:-1])

    for m in range(self.nnodes-1):
      t = t + dtsdc[m]

      qSDC[m+1] = qSDC[m] + dtsdc[m]*fSDC[0,m] + rhs[m]
      feval.evaluate(qSDC[m+1], t, fSDC[:,m+1], **kwargs)

      if gSDC is not None:
        fSDC[0,m+1] += gSDC[m+1]
