"""PyPFASST FEval class."""

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


class FEval(object):
    """Function evaluation base class.

    Methods in this class should be overridden with problem specific
    evaluators.  Each instance of the class is associated with a
    PFASST level.  The shape of the unknowns associated with this
    evaluator/level should be stored in the instance variable *shape*.
    The total number of unknowns associated with this evaluator/level
    should be stored in the instance variable *size* (note also that
    *size* should be the product of *shape*).  If the function is
    broken into several pieces (ie, explicit and pieces), then the
    number of pieces should be stored in the instance variable
    *pieces* (each piece has shape *shape* and size *size*).

    See also :py:class:`pfasst.explicit.ExplicitFEval` and
    :py:class:`pfasst.imex.IMEXFEval`.

    Attributes:

    .. attribute:: pieces

       Number of function evaluation pieces.  For example, for a
       purely explicit time stepper, this would 1, and for an
       implicit/explicit time stepper, this would be 2.

    .. attribute:: shape

       Shape of the unknowns associated with this evaluator.

    .. attribute:: size

       Number of unknown associated with this evaluator (this should
       be the product of the *shape* attribute).

    """

    def evaluate(self, y, t, f, **kwargs):
        """Evaluate function values *f(y, t)*.

        :param y: y (numpy array)
        :param t: time (float)
        :param f: result (numpy array)

        The (flattened) result is stored in *f*.

        **This method should be overridden.**

        By default, this sets *f* to 0.

        """

        f[...] = 0.0


