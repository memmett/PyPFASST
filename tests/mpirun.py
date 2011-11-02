
import os
import subprocess


def mpirun(args, nproc, sort=True):
    if 'VIRTUAL_ENV' in os.environ:
        python = os.environ['VIRTUAL_ENV'] + '/bin/python'
    else:
        python = 'python'

    cmd = [ 'mpirun', '-n', str(nproc), python ] + args
    cmd = ' '.join(cmd)

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    stdout, stderr = p.communicate()

    if sort:
        stdout = sorted(stdout.split('\n'))

    return stdout, stderr
