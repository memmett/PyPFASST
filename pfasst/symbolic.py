
import sympy

from sympy import Symbol, MatrixSymbol, Matrix

###############################################################################

class PFASST(object):

  levels = []
  level  = None

  @classmethod
  def add_level(self, level):
    PFASST.levels.append(level)

    if PFASST.level is None:
      PFASST.level = 0


class Level(object):

  def __init__(self, nnodes=9,
               R=None, T=None, F=None, SDC=None):

    self.level = len(PFASST.levels)
    self.nnodes = nnodes

    self.U = [ Symbol('U(%d,%d)' % (self.level, x)) for x in range(nnodes) ]

    if R is None:
      self.R = sympy.Function('R' + str(self.level))
    else:
      self.R = R

    if T is None:
      self.T = sympy.Function('T' + str(self.level))
    else:
      self.T = T

    if F is None:
      self.F = sympy.Function('F' + str(self.level))
    else:
      self.F = F

    if SDC is None:
      self.SDC = sympy.Function('SDC' + str(self.level))
    else:
      self.SDC = SDC


    PFASST.add_level(self)

  def __str__(self):
    return 'L' + str(self.level)


###############################################################################

def apply_operator(op, obj):

  if isinstance(obj, Level):
    obj = obj.U

  if isinstance(op, MatrixSymbol):
    return op * obj
  elif isinstance(op, Matrix):
    return op * obj

  return op(obj)


def apply_time_space_operator(op, obj):

  if isinstance(obj, Level):
    obj = obj.U

  if isinstance(obj, list):
    return [ apply_operator(op, x) for x in obj ]

  return op(obj)


def R(obj):
  """Apply current levels restriction operator to *obj*."""

  level = PFASST.levels[PFASST.level]
  PFASST.level += 1

  return apply_time_space_operator(level.R, obj)[::2]


def T(obj):
  """Apply current levels interpolation operator to *obj*."""

  level = PFASST.levels[PFASST.level]
  PFASST.level -= 1

  return apply_operator(level.T, obj)


def F(obj):
  """Apply current levels function operator to *obj*."""

  level = PFASST.levels[PFASST.level]
  PFASST.level -= 1

  return apply_operator(level.F, obj)


def SDC(obj):
  """Apply current levels SDC operator to *obj*."""

  level = PFASST.levels[PFASST.level]
  PFASST.level -= 1

  return apply_operator(level.SDC, obj)


###############################################################################

def fbeuler_sdc_operator(S, A, dt):
  """Build SDC sweep operator from SDC matrix *S* and function
  evaluation matrix *A*."""

  I  = 1
  U0 = 1

  return (I - dt*S*A)**(-1)*(U0 + dt*Stil)*A


