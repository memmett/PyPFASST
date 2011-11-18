'''PyPFASST Level class.'''

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


class Level(object):
  """PyPFASST level class.

  Attributes:

  .. attribute:: qSDC

     The unknown at each SDC node.

  .. attribute:: q0

     Initial condition (received from the previous time processor).

  .. attribute:: qend

     Final value (sent to the next time processor).

  .. attribute:: fSDC

     The function evaluations at each SDC node.

  .. attribute:: bSDC

     The RHS for SDC sweeps (the initial conditions and FAS
     corrections are used to form the RHS).

  .. attribute:: gSDC

     Time dependent forcing evaluations at each SDC node.

  .. attribute:: forcing

     True if time dependent forcing is present.

  See the :meth:`~pfasst.pfasst.PFASST.add_level` and
  :meth:`~pfasst.runner.Runner.allocate` methods for more
  details.
  """


  def __init__(self):

    super(Level, self).__init__()

    self.q0 = None              # initial condition
    self.qend = None            # end value
    self.bSDC = None            # right hand side for SDC sweeps
    self.qSDC = None            # unknown
    self.fSDC = None            # function evaluations
    self.gSDC = None            # forcing evaluations

    self.forcing = False        # no forcing by default

    self.qsend = None           # send buffer
    self.qrecv = None           # receive buffer

    self.send_request = None    # MPI send request
    self.recv_request = None    # MPI receive request


  #############################################################################

  def call_hooks(self, key, **kwargs):
    """Call hooks."""

    if key in self.hooks:
      for hook in self.hooks[key]:
        self.state.hook = key
        hook(self, self.state)


  #############################################################################

  def send(self, tag, blocking=False):
    """Send qend forward to the next time processor."""

    rank = self.mpi.rank

    self.call_hooks('pre-send')

    if blocking:
      self.mpi.comm.Send(
        self.qend, dest=self.mpi.forward, tag=tag)

    else:
      if self.send_request:
        self.send_request.Wait()

      self.qsend[:] = self.qend[:]
      self.send_request = self.mpi.comm.Isend(
        self.qsend, dest=self.mpi.forward, tag=tag)

    self.call_hooks('post-send')


  #############################################################################

  def post_receive(self, tag):
    """Post a receive request."""

    rank = self.mpi.rank

    self.recv_tag = tag
    self.recv_request = self.mpi.comm.Irecv(
      self.qrecv, source=self.mpi.backward, tag=tag)


  def receive(self, tag, blocking=False, ignore=False):
    """Receive q0 from the previous time processor."""

    rank = self.mpi.rank

    self.call_hooks('pre-receive')

    if blocking:
      self.mpi.comm.Recv(
        self.qrecv, source=self.mpi.backward, tag=tag)

    else:
      assert(tag == self.recv_tag)
      self.recv_request.Wait()

    if not ignore:
      self.q0[:] = self.qrecv[:]

    self.call_hooks('post-receive')



