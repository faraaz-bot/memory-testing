
from dataclasses import dataclass
from typing import *

import sympy

#
# Symbolic DFT matrix
#

re = sympy.re
im = sympy.im


def dft(n, direction=-1):
    '''Compute DFT matrix A such that X = A * x is the DFT of x.'''
    A = sympy.zeros(n, n)
    for i in range(n):
        for j in range(n):
            A[j,i] = sympy.exp(2*direction*sympy.pi*sympy.I * sympy.Rational(i*j, n))
    return A

#
# Code generator classes
#
# High level view: build a tree structure of the following classes.
# Each class requires a 'render' method that returns a valid C++
# string representation of itself.
#

@dataclass
class Value:
    '''Numeric (literal) value.'''
    value: Any
    def render(self) -> str:
        return '(' + str(self.value) + ')'


@dataclass
class Symbol:
    '''Variable.'''
    name: Any
    def render(self) -> str:
        return str(self.name)


class Complex:
    '''Complex variable, with .x and .y accessors.'''
    def __init__(self, name):
        self.name = name
        self.x = Symbol(self.name + '.x')
        self.y = Symbol(self.name + '.y')
    def render(self) -> str:
        return self.name


class LComplex:
    '''Literal complex value.'''
    def __init__(self, x, y):
        self.x, self.y = Value(x), Value(y)
    def render(self) -> str:
        return '{ ' + self.x.render() + ', ' + self.y.render() + ' }'


@dataclass
class Assign:
    '''Assignment (=) operator; lhs is normally a Symbol.'''
    lhs: Any
    rhs: Any
    def render(self) -> str:
        return self.lhs.render() + ' = ' + self.rhs.render()


@dataclass
class AssignAdd:
    '''Assiment (+=) operator.'''
    lhs: Any
    rhs: Any
    def render(self) -> str:
        return self.lhs.render() + ' += ' + self.rhs.render()


@dataclass
class Multiply:
    '''Multiply operator (list of operands).'''
    args: List[Any]
    def render(self) -> str:
        return ' * '.join([ x.render() for x in self.args ])


@dataclass
class Sum:
    '''Sum operator (list of operands).'''
    args: List[Any]
    def render(self) -> str:
        return ' + '.join([ x.render() for x in self.args ])


@dataclass
class Difference:
    '''Difference operator (list of operands).'''
    args: List[Any]
    def render(self) -> str:
        return ' - '.join([ x.render() for x in self.args ])


def render(stmts : List[Any]) -> str:
    '''Render helper: render list of statements and join with semicolons.'''
    return ';\n'.join([ s.render() for s in stmts ] + [''])


#
# Butterfly kernel generator: given DFT matrix A, uses symmetry
# properties to generate a butterly kernel.
#

def butterfly_(A):

    N = A.shape[0]
    R = [ Complex(f'(*R{i})') for i in range(N) ]
    x = [ Complex(f'x{i}') for i in range(N) ]
    dp, dm = Complex('dp'), Complex('dm')

    yield Assign(x[0], Sum(R))
    for i in range(1, N):
        yield Assign(x[i], R[0])

    for j in range(1, N//2+1):
        yield Assign(dp, Sum([R[j], R[N-j]]))
        yield Assign(dm, Difference([R[j], R[N-j]]))
        for i in range(1, N//2+1):
            alpha = LComplex(re(A[i,j]), im(A[i,j]))
            yield AssignAdd(x[i].x,
                                Difference([
                                    Multiply([alpha.x, dp.x]),
                                    Multiply([alpha.y, dm.y]) ]))
            yield AssignAdd(x[i].y,
                                Sum([
                                    Multiply([alpha.x, dp.y]),
                                    Multiply([alpha.y, dm.x]) ]))
            yield AssignAdd(x[N-i].x,
                                Sum([
                                    Multiply([alpha.x, dp.x]),
                                    Multiply([alpha.y, dm.y]) ]))
            yield AssignAdd(x[N-i].y,
                                Difference([
                                    Multiply([alpha.x, dp.y]),
                                    Multiply([alpha.y, dm.x]) ]))
    for i in range(N):
        yield Assign(R[i], x[i])


def butterfly(A):
    '''Generate statements to compute butterfly kernel given DFT matrix A.'''
    return list(butterfly_(A))


if __name__ == '__main__':
    A = dft(13, 1)
    print(render(butterfly(A.evalf(22))))
