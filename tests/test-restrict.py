'''PyPFASST restrict tests.'''

from itertools import product

import math
import numpy as np

import pfasst.imex
import pfasst.level

import linearad

from pfasst.restrict import restrict_time_space

# test these nvars and nodes...
nvars      = [ (64, 32), (512, 256) ]
nodes      = [ ('GL', 5, 3), ('CC', 9, 5), ('CC', 13, 7) ]

dt = 0.5


def restrict_fd_1d(yF, yG, **kwargs):
  """Finite-difference restrictor (1D)."""

  xrat = yF.shape[0] / yG.shape[0]
  yG[:] = yF[::xrat]


def test_restrict():

    for (nvar, node) in product(nvars, nodes):

        lF = linearad.LinearAD(nvar[0], Lx=1.0, nu=0.02, acst=5.0, t0=0.1)
        lG = linearad.LinearAD(nvar[1], Lx=1.0, nu=0.02, acst=5.0, t0=0.1)

        sdcF = pfasst.imex.IMEXSDC(node[0], node[1])
        sdcG = pfasst.imex.IMEXSDC(node[0], node[2])

        qF  = np.zeros((sdcF.nnodes,) + lF.shape)
        qG  = np.zeros((sdcG.nnodes,) + lG.shape)
        qGx = np.zeros((sdcG.nnodes,) + lG.shape)

        fF  = np.zeros((lF.pieces, sdcF.nnodes,) + lF.shape)
        fG  = np.zeros((lG.pieces, sdcG.nnodes,) + lG.shape)
        fGx = np.zeros((lG.pieces, sdcG.nnodes,) + lG.shape)

        # populate qF and fF
        for m in range(node[1]):
            t = dt * sdcF.nodes[m]
            lF.exact(t, qF[m,:])
            lF.f1_evaluate(qF[m,:], t, fF[0,m])
            lF.f2_evaluate(qF[m,:], t, fF[1,m])

        # populate qGx
        for m in range(node[2]):
            t = dt * sdcG.nodes[m]
            lG.exact(t, qGx[m,:])
            lG.f1_evaluate(qGx[m,:], t, fGx[0,m])
            lG.f2_evaluate(qGx[m,:], t, fGx[1,m])

        # restrict!
        F = pfasst.level.Level()
        F.restrict = restrict_fd_1d
        F.feval = lF

        G = pfasst.level.Level()
        G.feval = lG

        restrict_time_space(qF, qG, F, G)

        errq = np.log10(abs(qG - qGx).max())
        errf = np.log10(abs(fG - fGx).max())

        print 'nvar: %s, node: %s; log error: %lf, %lf' % (nvar, node, errq, errf)

        assert errq < -10


if __name__ == '__main__':
    test_restrict()










