'''1d linear AD equation.'''

import math
import numpy as np
import numpy.fft as fft

import pfasst.imex

class LinearAD(pfasst.imex.IMEXFEval):

    Lx   = 1.0
    nu   = 0.02
    acst = 1.0
    t0   = 1.0

    def __init__(self, size, Lx=1.0, acst=1.0, nu=0.02, t0=1.0, **kwargs):

        super(LinearAD, self).__init__()

        self.shape = (size,)
        self.size = size
        self.Lx   = Lx
        self.acst = acst
        self.nu   = nu
        self.t0   = t0

        self.wave_numbers = 2*math.pi/Lx * fft.fftfreq(size) * size
        self.laplacian    = -(2*math.pi/Lx * fft.fftfreq(size) * size)**2


    def f1_evaluate(self, y, t, f1, **kwargs):
        n = y.shape[0]

        z   = fft.fft(y)
        z_x = z * self.wave_numbers * 1j
        y_x = np.real(fft.ifft(z_x))

        f1[:] = -self.acst * y_x


    def f2_evaluate(self, y, t, f2, **kwargs):
        n = y.shape[0]

        z = fft.fft(y)
        op = self.nu * self.laplacian
        z = op * z

        f2[:] = np.real(fft.ifft(z))


    def f2_solve(self, rhs, y, t, dt, f2, **kwargs):
        n = y.shape[0]

        z = fft.fft(rhs)
        invop = 1.0 / (1.0 - self.nu*dt*self.laplacian)
        z = invop * z

        y[:] = np.real(fft.ifft(z))

        op = self.nu * self.laplacian
        z = op * z

        f2[:] = np.real(fft.ifft(z))


    def exact(self, t, q):

        size = self.size
        Lx   = self.Lx
        acst = self.acst
        nu   = self.nu
        t0   = self.t0

        q[:] = 0.0

        for ii in range(-5, 6):
            for i in range(size):
                x = Lx*(i-size/2)/size + ii*Lx - acst*t
                q[i] += (4.0*math.pi*nu*(t+t0))**(-0.5) * math.exp(-x**2/(4.0*nu*(t+t0)))

