Download
========

The source code for PyPFASST is hosted on GitHub.  Please see the
`PyPFASST project page`_ to browse the code, submit bug reports,
contribute etc.

To download the latest development version of PyPFASST using git::

$ git clone git://github.com/memmett/PyPFASST.git

If you have already done this and just need to grab the latest
version::

$ cd PyPFASST
$ git pull

If you don't have git, you can download the latest version of PyPFASST
as a

* tarball: `<https://github.com/memmett/PyPFASST/tarball/master>`_
* zipball: `<https://github.com/memmett/PyPFASST/zipball/master>`_.

Installation
------------

PyPFASST requires two Python packages: `mpi4py`_ and `NumPy`_; and an
MPI library.  To install these packages on Debian or Ubuntu::

$ sudo apt-get install python-dev python-numpy mpich2 libmpich2-dev

Next, you need to install mpi4py.  The easiest way to do this is with
PIP::

$ sudo apt-get install python-pip
$ sudo pip install mpi4py


.. _`PyPFASST project page`: https://github.com/memmett/PyPFASST
.. _`mpi4py`: http://mpi4py.scipy.org/
.. _`NumPy`: http://www.scipy.org/
