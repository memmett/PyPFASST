'''PyPFASST interpolate tests.'''

from itertools import product

import math
import numpy as np
import numpy.fft as fft

import pfasst.imex
import pfasst.level

import linearad

from pfasst.interpolate import interpolate_correction_time_space, interpolate_correction
from pfasst.interpolate import time_interpolation_matrix

# test these nvars and nodes...
nvars      = [ (64, 32), (256, 128), (512, 256) ]
nodes      = [ ('GL', 5, 3), ('CC', 9, 5), ('CC', 13, 7) ]


dt = 1.0/128


def restrict_fd_1d(yF, yG, **kwargs):
  """Finite-difference restrictor (1D)."""

  xrat = yF.shape[0] / yG.shape[0]
  yG[:] = yF[::xrat]



def interpolate(yF, yG, **kwargs):

    N = yG.shape[0]

    zG = fft.fft(yG)
    zF = np.zeros(2*N, zG.dtype)

    zF[:N/2] = zG[:N/2]
    zF[-N/2+1:] = zG[-N/2+1:]

    yF[:] = np.real(2*fft.ifft(zF))


def test_interpolate_correction_time_space():

    for (nvar, node) in product(nvars, nodes):

        lF = linearad.LinearAD(nvar[0], Lx=1.0, nu=0.02, acst=5.0, t0=0.1)
        lG = linearad.LinearAD(nvar[1], Lx=1.0, nu=0.02, acst=5.0, t0=0.1)

        sdcF = pfasst.sdc.SDC(node[0], node[1])
        sdcG = pfasst.sdc.SDC(node[0], node[2])

        qF  = np.zeros((sdcF.nnodes,) + lF.shape)
        qG  = np.zeros((sdcG.nnodes,) + lG.shape)
        qFx = np.zeros((sdcF.nnodes,) + lF.shape)

        fF  = np.zeros((lF.pieces, sdcF.nnodes,) + lF.shape)
        fG  = np.zeros((lG.pieces, sdcG.nnodes,) + lG.shape)
        fFx = np.zeros((lF.pieces, sdcF.nnodes,) + lF.shape)

        # populate qG and fG
        for m in range(node[2]):
            t = dt * sdcG.nodes[m]
            lG.exact(t, qG[m,:])
            lG.f1_evaluate(qG[m,:], t, fG[0,m])
            lG.f2_evaluate(qG[m,:], t, fG[1,m])

        # populate qFx
        for m in range(node[1]):
            t = dt * sdcF.nodes[m]
            lF.exact(t, qFx[m,:])
            lF.f1_evaluate(qFx[m,:], t, fFx[0,m])
            lF.f2_evaluate(qFx[m,:], t, fFx[1,m])

        # get matrix
        time_interp_mat = time_interpolation_matrix(sdcF.nodes, sdcG.nodes)

        # interpolate!
        F = pfasst.level.Level()
        F.restrict = restrict_fd_1d
        F.interpolate = interpolate
        F.time_interp_mat = time_interp_mat
        F.feval = lF
        F.sdc = sdcF

        G = pfasst.level.Level()
        G.feval = lG
        G.sdc = sdcG

        interpolate_correction_time_space(qF, qG, F, G)

        errq = np.log10(abs(qF - qFx).max())
        errf = np.log10(abs(fF - fFx).max())

        print 'nvar: %s, node: %s; log error: %lf, %lf' % (nvar, node, errq, errf)


if __name__ == '__main__':
    np.set_printoptions(linewidth=100)
    test_interpolate_correction_time_space()
