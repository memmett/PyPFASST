'''PyPFASST '''

import numpy as np
from mpi4py import MPI

import pfasst.mpi

mpi = pfasst.mpi.PFASSTMPI()
mpi.create_simple_communicators(4, 4)

world = MPI.COMM_WORLD
world_rank = world.rank
color = mpi.rank
rank  = mpi.space.rank

print 'created communicators: c%02d w%02d|' % (color, world_rank)

message = np.array([color, rank, world_rank])
new_message = np.empty(message.shape, dtype=message.dtype)
i_am = message

if color == 0:

  msg = "sending   (%s): %s|" % (i_am, message)
  print msg

  world.Send(message, dest=mpi.forward, tag=10*rank+color+1)

elif color < 3:

  world.Recv(new_message, source=mpi.backward, tag=10*rank+color)

  msg = "received  (%s): %s|" % (i_am, new_message)
  print msg

  msg = "sending   (%s): %s|" % (i_am, message)
  print msg

  world.Send(message, dest=mpi.forward, tag=10*rank+color+1)

else:

  world.Recv(new_message, source=mpi.backward, tag=10*rank+color)

  msg = "received  (%s): %s|" % (i_am, new_message)
  print msg
