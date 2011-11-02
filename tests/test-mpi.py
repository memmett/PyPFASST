import os
import subprocess

ok = '''
created communicators: c00 w00
created communicators: c00 w04
created communicators: c00 w08
created communicators: c00 w12
created communicators: c01 w01
created communicators: c01 w05
created communicators: c01 w09
created communicators: c01 w13
created communicators: c02 w02
created communicators: c02 w06
created communicators: c02 w10
created communicators: c02 w14
created communicators: c03 w03
created communicators: c03 w07
created communicators: c03 w11
created communicators: c03 w15
received  ([ 1  3 13]): [ 0  3 12]
received  ([ 2  2 10]): [1 2 9]
received  ([ 2  3 14]): [ 1  3 13]
received  ([ 3  2 11]): [ 2  2 10]
received  ([ 3  3 15]): [ 2  3 14]
received  ([1 0 1]): [0 0 0]
received  ([1 1 5]): [0 1 4]
received  ([1 2 9]): [0 2 8]
received  ([2 0 2]): [1 0 1]
received  ([2 1 6]): [1 1 5]
received  ([3 0 3]): [2 0 2]
received  ([3 1 7]): [2 1 6]
sending   ([ 0  3 12]): [ 0  3 12]
sending   ([ 1  3 13]): [ 1  3 13]
sending   ([ 2  2 10]): [ 2  2 10]
sending   ([ 2  3 14]): [ 2  3 14]
sending   ([0 0 0]): [0 0 0]
sending   ([0 1 4]): [0 1 4]
sending   ([0 2 8]): [0 2 8]
sending   ([1 0 1]): [1 0 1]
sending   ([1 1 5]): [1 1 5]
sending   ([1 2 9]): [1 2 9]
sending   ([2 0 2]): [2 0 2]
sending   ([2 1 6]): [2 1 6]'''


def test_simple_communicator():

    if os.path.exists('mpi.py'):
        p = subprocess.Popen('mpirun -n 16 python mpi.py',
                             stdout=subprocess.PIPE,
                             shell=True)
    else:
        p = subprocess.Popen('mpirun -n 16 python tests/mpi.py',
                             stdout=subprocess.PIPE,
                             shell=True)

    (stdout, stderr) = p.communicate()

    stdout = stdout.split('\n')
    stdout = ''.join(stdout)
    stdout = stdout.split('|')
    stdout = map(lambda x: x.strip('|'), stdout)
    stdout = "\n".join(sorted(stdout))

    print 'Expected: '
    print ok

    print 'Received: '
    print stdout

    assert(stdout == ok)


if __name__ == '__main__':

    test_simple_communicator()

