
from mpirun import mpirun


def test_advection_1d():
    args = [ 'examples/advection/main.py' ]
    stdout, stderr = mpirun(args, 8)

    last_error = float(stdout[-1].split()[-1])
    print 'PFASST 1d log10(error):', last_error

    assert(last_error < -12)


def test_advection_rk():
    args = [ 'examples/advection/main.py', '-R', '-l', '1' ]
    stdout, stderr = mpirun(args, 1)

    last_error = float(stdout[-1].split()[-1])
    print 'RK4 log10(error):', last_error

    assert(last_error < -6)


# def test_advection_2d():
#     args = [ 'examples/advection/main.py', '-d', '2' ]
#     stdout, stderr = mpirun(args, 8)

#     last_error = float(stdout[-1].split()[-1])
#     print '2d PFASST log10(error):', last_error

#     assert(last_error < -10)


if __name__ == '__main__':
    test_advection_1d()
    test_advection_rk()
    #test_advection_2d()
