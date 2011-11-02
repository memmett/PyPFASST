'''PyPFASST SDC tests.'''

import math
import numpy as np

import pfasst.imex

import linearad

# test these nodes...

nodes      = [ ('GL', 3), ('GL', 5), ('GL', 9), ('GL', 13) ]
tolerances = [        -7,       -13,       -13,        -13 ]
sweeps     = 12

# test problem
size  = 128
feval = linearad.LinearAD(size, Lx=1.0, nu=0.02, acst=5.0, t0=0.1)
dt    = 0.0007


def test_sdc():

    for i, (qtype, nnodes) in enumerate(nodes):

        # set initial condition
        q0 = np.zeros(size)
        b  = np.zeros((nnodes, size))
        feval.exact(0.0, q0)

        q1 = np.zeros(size)
        feval.exact(dt, q1)

        qSDC = np.zeros((nnodes, size))
        fSDC = np.zeros((2, nnodes, size))

        # spread and eval
        for m in range(nnodes):
            qSDC[m] = q0
            feval.f1_evaluate(q0, 0.0, fSDC[0,m])
            feval.f2_evaluate(q0, 0.0, fSDC[1,m])

        # create sdc object
        sdc = pfasst.imex.IMEXSDC(qtype, nnodes)

        # sweep and print error
        for s in range(sweeps):
            b[0] = q0
            sdc.sweep(b, 0.0, dt, qSDC, fSDC, feval)
            q2 = qSDC[-1]

            err = np.log10(abs(q1-q2).max())

            print 'node: %s; sweep %d; log error: %lf' % ((qtype, nnodes), s+1, err)

        assert(err < tolerances[i])


if __name__ == '__main__':
    test_sdc()
