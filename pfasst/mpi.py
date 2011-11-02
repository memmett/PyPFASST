'''PyPFASST MPI helpers (feel free to contribe more!).'''

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


class PFASSTMPI(object):
  """PFASST MPI helpers.

  Attributes:

  .. attribute:: nspace

     Number of spatial processors.

  .. attribute:: ntime

     Number of time processors.

  .. attribute:: space

     Spatial MPI communicator.

  .. attribute:: comm

     Time MPI communicator.

  .. attribute:: forward

     Forward in time MPI rank.

  .. attribute:: backward

     Backward in time MPI rank.

  .. attribute:: rank

     Time MPI rank.

  """

  def create_simple_communicators(self, nspace, ntime, comm=MPI.COMM_WORLD):
    """Create MPI communicators using simple colouring.

    :param nspace: number of spatial processors
    :param ntime: number of time processors
    :param comm: base communicator (defaults to MPI.COMM_WORLD)

    Creates MPI communicators so that there are *ntime* groups of
    *nspace* processors.  Within each spatial group, the proceccors are
    intraconnected through the ``space`` communicator.  Each processor
    in a spatial group is connected to its equivalent processors (ie,
    same spatial rank) in time through the ``forward`` and ``backward``
    source/destination rank.

    The above is summarised by::

      color 0:        rank 0  <-- space --> ... <-- space -->  rank nspace-1
                                    |                 ^
                                    |                 |
                                 forward           backward
                                    |                 |
                                    v                 |
      color ntime-1:  rank 0  <-- space --> ... <-- space -->  rank nspace-1

    Finally, the ``forward`` and ``backward`` ranks are cyclic.  That
    is, a message send forward from color ``ntime-1`` is received by
    color ``0``, and visa versa.

    """

    size = comm.size
    rank = comm.rank

    if size != ntime*nspace:
      raise ValueError('processor number mismatch: '
                       'was expecting ntime*nspace '
                       '(= %d*%d = %d), ' % (ntime, nspace, ntime*nspace),
                       'but received only %d' % (size))

    if nspace == 1:
      color = rank
      space = MPI.COMM_SELF
    else:
      color = rank % ntime
      space = MPI.Intracomm.Split(comm, color)

    forward  = rank + 1
    backward = rank - 1

    if color == ntime-1:
      forward = (rank/ntime)*ntime

    if color == 0:
      backward = rank + ntime - 1


    self.ntime    = ntime
    self.nspace   = nspace
    self.space    = space
    self.forward  = forward
    self.backward = backward
    self.rank     = color
    self.comm     = comm
