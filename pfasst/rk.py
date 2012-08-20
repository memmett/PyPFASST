"""PyPFASST serial ARK class."""

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

from runner import Runner


class ARKRunner(Runner):
  """Use the semi-implicit RK scheme of Carpenter and Kennedy instead of SDC.

  Serial only!
  """

  def __init__(self, order):

    self.order = order

    if order == 4:
      be = np.zeros(6)
      ae = np.zeros((6,6))
      ai = np.zeros((6,6))

      be[0] = 82889.0/524892.0
      be[1] = 0.0
      be[2] = 15625.0/83664.0
      be[3] = 69875.0/102672.0
      be[4] =-2260.0/8211.0
      be[5] = 1.0/4.0

      ae[1,0] = 1.0/2.0
      ae[2,0] = 13861.0/62500.0
      ae[2,1] = 6889.0/62500.0
      ae[3,0] =-116923316275.0/2393684061468.0
      ae[3,1] =-2731218467317.0/15368042101831.0
      ae[3,2] = 9408046702089.0/11113171139209.0
      ae[4,0] =-451086348788.0/2902428689909.0
      ae[4,1] =-2682348792572.0/7519795681897.0
      ae[4,2] = 12662868775082.0/11960479115383.0
      ae[4,3] = 3355817975965.0/11060851509271.0
      ae[5,0] = 647845179188.0/3216320057751.0
      ae[5,1] = 73281519250.0/8382639484533.0
      ae[5,2] = 552539513391.0/3454668386233.0
      ae[5,3] = 3354512671639.0/8306763924573.0
      ae[5,4] = 4040.0/17871.0

      ai[1,0] = 1.0/4.0
      ai[1,1] = 1.0/4.0
      ai[2,0] = 8611.0/62500.0
      ai[2,1] =-1743.0/31250.0
      ai[2,2] = 1.0/4.0
      ai[3,0] = 5012029.0/34652500.0
      ai[3,1] =-654441.0/2922500.0
      ai[3,2] = 174375.0/388108.0
      ai[3,3] = 1.0/4.0
      ai[4,0] = 15267082809.0/155376265600.0
      ai[4,1] =-71443401.0/120774400.0
      ai[4,2] = 730878875.0/902184768.0
      ai[4,3] = 2285395.0/8070912.0
      ai[4,4] = 1.0/4.0
      ai[5,0] = 82889.0/524892.0
      ai[5,1] = 0.0
      ai[5,2] = 15625.0/83664.0
      ai[5,3] = 69875.0/102672.0
      ai[5,4] =-2260.0/8211.0
      ai[5,5] = 1.0/4.0

    elif order == 5:

      be = np.zeros(8)
      ae = np.zeros((8,8))
      ai = np.zeros((8,8))

      be[0] =-872700587467.0/9133579230613.0
      be[1] = 0.0
      be[2] = 0.0
      be[3] = 24752842989968.0/10584050607295.0
      be[4] =-1143369518992.0/8141816002931.0
      be[5] =-13732001328083.0/6631934148607.0
      be[6] = 31972909776967.0/41911059379164.0
      be[7] = 41.0/200.0

      ae[1,0] = 41.0/100.0
      ae[2,0] = 367902744464.0/2072280473677.0
      ae[2,1] = 677623207551.0/8224143866563.0
      ae[3,0] = 1268023523408.0/10340822734521.0
      ae[3,1] = 0.0
      ae[3,2] = 1029933939417.0/13636558850479.0
      ae[4,0] = 29921928531207.0/13065330778082.0
      ae[4,1] = 0.0
      ae[4,2] = 115140034464727.0/10239288090423.0
      ae[4,3] =-78522360150645.0/6224472171947.0
      ae[5,0] = 4175963610463.0/10363619370981.0
      ae[5,1] = 0.0
      ae[5,2] = 5941611906955.0/4388151832759.0
      ae[5,3] =-14081064728885.0/9477725119419.0
      ae[5,4] =-146841095096.0/4698013173029.0
      ae[6,0] = 20649979511554.0/14103843532755.0
      ae[6,1] = 0.0
      ae[6,2] = 46104976391489.0/6376485181372.0
      ae[6,3] =-68205481673867.0/8694569480018.0
      ae[6,4] =-1.0/8.0
      ae[6,5] =-1.0/8.0
      ae[7,0] =-22436580330729.0/13396508891632.0
      ae[7,1] = 0.0
      ae[7,2] =-61519777358797.0/9628354034130.0
      ae[7,3] = 133952912771311.0/9117280366678.0
      ae[7,4] = 819112427236.0/8652635578785.0
      ae[7,5] =-87740800058441.0/12167367327014.0
      ae[7,6] = 9714094484631.0/6525933883406.0

      ai[1,0] = 41.0/200.0
      ai[1,1] = 41.0/200.0
      ai[2,0] = 41.0/400.0
      ai[2,1] =-567603406766.0/11931857230679.0
      ai[2,2] = 41.0/200.0
      ai[3,0] = 683785636431.0/9252920307686.0
      ai[3,1] = 0.0
      ai[3,2] =-110385047103.0/1367015193373.0
      ai[3,3] = 41.0/200.0
      ai[4,0] = 3435385757185.0/11481209014384.0
      ai[4,1] = 0.0
      ai[4,2] = 11157427131511.0/4528506187550.0
      ai[4,3] =-9556729888537.0/4666283647179.0
      ai[4,4] = 41.0/200.0
      ai[5,0] = 218866479029.0/1489978393911.0
      ai[5,1] = 0.0
      ai[5,2] = 638256894668.0/5436446318841.0
      ai[5,3] =-2405872765194.0/10851833147315.0
      ai[5,4] =-60928119172.0/8023461067671.0
      ai[5,5] = 41.0/200.0
      ai[6,0] = 1020004230633.0/5715676835656.0
      ai[6,1] = 0.0
      ai[6,2] = 27712922358947.0/27176279295670.0
      ai[6,3] =-1316382662167.0/5941820308240.0
      ai[6,4] =-211217309593.0/5846859502534.0
      ai[6,5] =-4710367281160.0/8634419175717.0
      ai[6,6] = 41.0/200.0
      ai[7,0] =-872700587467.0/9133579230613.0
      ai[7,1] = 0.0
      ai[7,2] = 0.0
      ai[7,3] = 24752842989968.0/10584050607295.0
      ai[7,4] =-1143369518992.0/8141816002931.0
      ai[7,5] =-13732001328083.0/6631934148607.0
      ai[7,6] = 31972909776967.0/41911059379164.0
      ai[7,7] = 41.0/200.0

    else:
      raise NotImplementedError('RK order %d not implemented yet.  Try 4 or 5.' % order)

    self.be = be
    self.ae = ae
    self.ai = ai


  #############################################################################

  def allocate(self, dtype):

    level = self.levels[0]

    shape  = level.feval.shape
    pieces = level.feval.pieces

    if self.order == 4:
      stages = 6
    elif self.order == 5:
      stages = 8

    assert(self.mpi.comm.size == 1)
    assert(pieces == 2)

    level.q0    = np.zeros(shape, dtype=dtype)
    level.qend  = np.zeros(shape, dtype=dtype)
    level.qRK   = np.zeros((stages,)+shape, dtype=dtype)
    level.fRK   = np.zeros((pieces,stages,)+shape, dtype=dtype)
    level.rhs   = np.zeros(shape, dtype=dtype)

    level.qSDC = None
    level.fSDC = None


  #############################################################################

  def run(self, u0, dt, tend, iterations, **kwargs):
    """Run in serial (RK)."""

    #### short cuts, state, options

    F = self.levels[0]

    self.state.dt   = dt
    self.state.tend = tend

    be = self.be
    ae = self.ae
    ai = self.ai


    #### set initial condition

    try:
      F.q0[...] = u0
    except ValueError:
      raise ValueError, 'initial condition shape mismatch'


    #### time step loop

    if self.order == 4:
      stages = 6
    elif self.order == 5:
      stages = 8

    nblocks = int(math.ceil(tend/dt))

    for block in range(nblocks):

      t0 = dt*block

      self.state.t0     = t0
      self.state.block  = block
      self.state.step   = block

      F.call_hooks('pre-sweep', **kwargs)

      # set initial condtion and evaluate at finest level
      F.qRK[0] = F.q0
      F.feval.evaluate(F.q0, t0, F.fRK[:,0])

      rhs = np.zeros(F.feval.shape, dtype=F.q0.dtype)

      # stage loop
      for k in range(1,stages):

        # build rhs
        rhs[...] = F.qRK[0]
        for j in range(k):
          rhs += dt * ( ae[k,j]*F.fRK[0,j] + ai[k,j]*F.fRK[1,j] )

        # evaluate
        # XXX: the t0's are incorrrect
        F.feval.f2_solve(rhs, F.qRK[k], t0, dt*ai[k,k], F.fRK[1,k])
        F.feval.f1_evaluate(F.qRK[k], t0, F.fRK[0,k])


      # sum stages to get end value
      F.qend[...] = F.qRK[0]
      for j in range(stages):
        F.qend += dt * be[j] * ( F.fRK[0,j] + F.fRK[1,j] )

      F.call_hooks('post-sweep', **kwargs)
      F.call_hooks('end-step', **kwargs)

      # set next initial condition
      F.q0[...] = F.qend

    F.call_hooks('end', **kwargs)
