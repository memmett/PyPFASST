"""Solve various advection/diffusion type equations with PyPFASST."""

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
import numpy.fft as fft

import pfasst.imex

from pfpack.interpolators import spectral_periodic_masks


###############################################################################
# define AD level

class AD(pfasst.imex.IMEXFEval):
  """IMEX FEval class for the adv/diff equation (or viscous Burgers)."""

  def __init__(self, shape=None, refinement=1,
               dim=1, L=1.0, nu=0.005, v=1.0, t0=1.0,
               burgers=False, verbose=False, **kwargs):

    super(AD, self).__init__()

    N = shape[0] / refinement

    self.shape = dim*(N,)
    self.size  = N**dim
    self.N     = N
    self.L     = L
    self.v     = v
    self.nu    = nu
    self.t0    = t0
    self.dim   = dim
    self.burgers = burgers


    # frequencies = 2*pi/L * (wave numbers)
    K = 2*math.pi/L * fft.fftfreq(N) * N

    if verbose:
      print 'building operators...'

    # operators
    if dim == 1:

      sgradient = K*1j
      laplacian = -K**2

    elif dim == 2:

      sgradient = np.zeros(self.shape, dtype=np.complex128)
      laplacian = np.zeros(self.shape, dtype=np.complex128)

      for i in xrange(N):
        for j in xrange(N):
          laplacian[i,j] = -(K[i]**2 + K[j]**2)
          sgradient[i,j] = K[i] * 1j + K[j] * 1j

    elif dim == 3:

      sgradient = np.zeros(self.shape, dtype=np.complex128)
      laplacian = np.zeros(self.shape, dtype=np.complex128)

      for i in range(N):
        for j in range(N):
          for k in range(N):
            laplacian[i,j,k] = -(
              K[i]**2 + K[j]**2 + K[k]**2)
            sgradient[i,j,k] = (
              K[i] * 1j + K[j] * 1j + K[k] * 1j)

    else:
      raise ValueError, 'dimension must be 1, 2, or 3'

    self.sgradient  = sgradient
    self.laplacian  = laplacian

    # spectral interpolation masks
    if verbose:
      print 'building interpolation masks...'

    self.full, self.half = spectral_periodic_masks(dim, N)


  def f1_evaluate(self, u, t, f1, **kwargs):
    """Evaluate explicit piece."""

    z = fft.fftn(u)

    z_sgrad = self.sgradient * z
    u_sgrad = np.real(fft.ifftn(z_sgrad))

    if self.burgers:
      f1[...] = - (u * u_sgrad)
    else:
      f1[...] = - (self.v * u_sgrad)


  def f2_evaluate(self, y, t, f2, **kwargs):
    """Evaluate implicit piece."""

    z = fft.fftn(y)
    z = self.nu * self.laplacian * z
    u = np.real(fft.ifftn(z))

    f2[...] = u


  def f2_solve(self, rhs, y, t, dt, f2, **kwargs):
    """Solve and evaluate implicit piece."""

    # solve (rebuild operator every time, as dt may change)
    invop = 1.0 / (1.0 - self.nu*dt*self.laplacian)

    z = fft.fftn(rhs)
    z = invop * z

    y[...] = np.real(fft.ifftn(z))

    # evaluate
    z = self.nu * self.laplacian * z

    f2[...] = np.real(fft.ifftn(z))


  def exact(self, t, q):
    """Exact solution (for adv/diff equation)."""

    if self.burgers:
      raise ValueError, 'exact solution not known for non-linear case'

    dim  = self.dim
    L    = self.L
    v    = self.v
    nu   = self.nu
    t0   = self.t0

    q1 = np.zeros(self.shape, q.dtype)

    images = range(-1,2)
    #images = [0]

    if dim == 1:

      nx, = self.shape

      for ii in images:
        for i in range(nx):
          x = L*(i-nx/2)/nx + ii*L - v*t
          q1[i] += ( (4.0*math.pi*nu*(t+t0))**(-0.5)
                     * math.exp(-x**2/(4.0*nu*(t+t0))) )

    elif dim == 2:

      nx, ny = self.shape

      for ii in images:
        for jj in images:

          for i in range(nx):
            x = L*(i-nx/2)/nx + ii*L - v*t
            for j in range(ny):
              y = L*(j-ny/2)/ny + jj*L - v*t

              q1[i,j] += ( (4.0*math.pi*nu*(t+t0))**(-1.0)
                           * math.exp(-(x**2+y**2)/(4.0*nu*(t+t0))) )


    elif dim == 3:

      nx, ny, nz = self.shape

      for ii in images:
        for jj in images:
          for kk in images:

            for i in range(nx):
              x = L*(i-nx/2)/nx + ii*L - v*t
              for j in range(ny):
                y = L*(j-ny/2)/ny + jj*L - v*t
                for k in range(nz):
                  z = L*(j-nz/2)/nz + kk*L - v*t

                  q1[i,j,k] += ( (4.0*math.pi*nu*(t+t0))**(-1.5)
                                 * math.exp(-(x**2+y**2+z**2)/(4.0*nu*(t+t0))) )


    q[...] = q1


###############################################################################
# define interpolator and restrictor

def interpolate(yF, yG, fevalF=None, fevalG=None,
                dim=1, xrat=2, interpolation_order=-1, **kwargs):
  """Interpolate yG to yF."""

  if interpolation_order == -1:

    zG = fft.fftn(yG)
    zF = np.zeros(fevalF.shape, zG.dtype)

    zF[fevalF.half] = zG[fevalG.full]

    yF[...] = np.real(2**dim*fft.ifftn(zF))

  elif interpolation_order == 2:

    if dim != 1:
      raise NotImplementedError

    yF[0::xrat] = yG
    yF[1::xrat] = (yG + np.roll(yG, -1)) / 2.0

  elif interpolation_order == 4:

    if dim != 1:
      raise NotImplementedError

    yF[0::xrat] = yG
    yF[1::xrat] = ( - np.roll(yG,1)
                    + 9.0*yG
                    + 9.0*np.roll(yG,-1)
                    - np.roll(yG,-2) ) / 16.0

  else:
    raise ValueError, 'interpolation order must be -1, 2 or 4'


def restrict(yF, yG, fevalF=None, fevalG=None,
             dim=1, xrat=2, **kwargs):
  """Restrict yF to yG."""

  if yF.shape == yG.shape:
    yG[:] = yF[:]

  elif dim == 1:
    yG[:] = yF[::xrat]

  elif dim == 2:
    y = np.reshape(yF, fevalF.shape)
    yG[...] = y[::xrat,::xrat]

  elif dim == 3:
    y = np.reshape(yF, fevalF.shape)
    yG[...] = y[::xrat,::xrat,::xrat]
