"""Solve the advection/diffusion equation with PyPFASST."""

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


from mpi4py import MPI

import argparse

import pfasst
import pfasst.imex

from ad import *


######################################################################
# options

parser = argparse.ArgumentParser(
    description='solve the advection/diffusion equation')
parser.add_argument('-d',
                    type=int,
                    dest='dim',
                    default=1,
                    help='number of dimensions, defaults to 1')
parser.add_argument('-n',
                    type=int,
                    dest='steps',
                    default=MPI.COMM_WORLD.size,
                    help='number of time steps, defaults to number of mpi processes')
parser.add_argument('-l',
                    type=int,
                    dest='nlevs',
                    default=3,
                    help='number of levels, defaults to 3')
parser.add_argument('-R',
                    dest='RK',
                    default=False,
                    action='store_true',
                    help='use serial Runge-Kutta integrator')
options = parser.parse_args()


###############################################################################
# config

comm  = MPI.COMM_WORLD
nproc = comm.size

dt   = 0.01
tend = dt*options.steps

N = 1024
D = options.dim

nnodes = [ 9, 5, 3 ]


###############################################################################
# init pfasst

pf = pfasst.PFASST()
pf.simple_communicators(ntime=nproc, comm=comm)

for l in range(options.nlevs):
  F = AD(shape=D*(N,), refinement=2**l, dim=D)
  SDC = pfasst.imex.IMEXSDC('GL', nnodes[l])
  pf.add_level(F, SDC, interpolate, restrict)

if len(pf.levels) > 1:
  pf.levels[-1].sweeps = 2


###############################################################################
# add hooks

def echo_error(level, state, **kwargs):
  """Compute and print error based on exact solution."""

  if level.feval.burgers:
    return

  y1 = np.zeros(level.feval.shape)
  level.feval.exact(state.t0+state.dt, y1)

  err = np.log10(abs(level.qend-y1).max())

  print 'step: %03d, iteration: %03d, position: %d, level: %02d, error: %f' % (
    state.step, state.iteration, state.cycle, level.level, err)

pf.add_hook(0, 'post-sweep', echo_error)


###############################################################################
# create initial condition and run

F  = AD(shape=D*(N,), dim=D)
q0 = np.zeros(F.shape)
F.exact(0.0, q0)

pf.run(q0=q0, dt=dt, tend=tend, RK=(4 if options.RK else None))
