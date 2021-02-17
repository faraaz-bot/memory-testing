'''HIP code generator.'''

import subprocess

from pathlib import Path as path
from typing import *

#
# Helpers
#

def join(sep, n):
    if isinstance(n, BaseNode):
        return sep.join([ str(x) for x in n.args ])
    return sep.join([ str(x) for x in n ])


def declarations(xs):
    return [ x.declaration() for x in xs ]


def format(code):
    p = subprocess.run(['clang-format-10', '-style=file'], stdout=subprocess.PIPE, input=str(code), encoding='ascii', check=True)
    return p.stdout


def format_and_write(fname, code):
    code = format(code)
    f = path(fname)
    if f.exists():
        existing = f.read_text()
        if existing == code:
            return
    f.write_text(code)


def walk(x):
    if isinstance(x, BaseNode):
        yield x
        for a in x.args:
            if isinstance(a, BaseNode):
                yield from a
            else:
                yield a

def depth_first(x, f):
    if isinstance(x, BaseNode):
        y = type(x)()
        y.args = [ depth_first(a, f) for a in x.args ]
        return f(y)
    return f(x)


#
# Code generator base classes
#

def name_args(names):
    def name_args_decorator(target):
        for i, name in enumerate(names):
            setattr(target, name, property(lambda self, idx=i: self.args[idx]))
        def new_init(self, *args, **kwargs):
            self.args = [ None for x in names ]
            for i, arg in enumerate(args):
                self.args[i] = arg
            for i, name in enumerate(names):
                if name in kwargs:
                    self.args[i] = kwargs[name]
        target.__init__ = new_init
        return target
    return name_args_decorator


class BaseNode:
    args: List[Any]
    kwargs = None
    sep: str = None
    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        if hasattr(self, '__post_init__'):
            getattr(self, '__post_init__')(self)
    def __str__(self):
        if self.sep is not None:
            return self.sep.join([str(x) for x in self.args])
        return str(self.args[0])
    def __iter__(self):
        return walk(self)


class BaseNodeOps(BaseNode):
    def __add__(self, a):
        return Add(self, a)
    def __radd__(self, a):
        return Add(a, self)
    def __sub__(self, a):
        return Sub(self, a)
    def __rsub__(self, a):
        return Sub(a, self)
    def __mul__(self, a):
        return Multiply(self, a)
    def __rmul__(self, a):
        return Multiply(a, self)
    def __mod__(self, a):
        return Mod(self, a)
    def __rmod__(self, a):
        return Mod(a, self)
    def __truediv__(self, a):
        return Divide(self, a)
    def __rtruediv__(self, a):
        return Divide(a, self)
    def __ge__(self, a):
        return GreaterEqual(self, a)
    def __gt__(self, a):
        return Greater(self, a)
    def __le__(self, a):
        return LessEqual(self, a)
    def __lt__(self, a):
        return Less(self, a)


class ArgumentList(BaseNode):
    def __str__(self):
        args = []
        for x in self.args:
            if isinstance(x, Variable):
                args.append(x.argument())
            else:
                args.append(str(x))
        return ', '.join(args)


class StatementList(BaseNode):
    def __add__(self, lst):
        if isinstance(lst, list):
            self.args.extend(lst)
        elif isinstance(lst, StatementList):
            self.args.extend(lst.args)
        else:
            self.args.append(lst)
        return self


class Declaration(BaseNode):
    pass


class String(BaseNode):
    pass


class TemplateList(ArgumentList):
    pass


#
# Operators
#

def make_unary(prefix):
    def decorator(target):
        target.__str__ = lambda self: prefix + self.args[0]
        return target
    return decorator


def make_binary(separator):
    def decorator(target):
        target.sep = separator
        return target
    return decorator


@make_unary('&')
class Address(BaseNode):
    pass

@name_args(['lhs', 'rhs'])
class Assign(BaseNode):
    def __str__(self):
        return str(self.args[0]) + ' = ' + str(self.args[1]) + ';'


@make_binary('.')
class Component(BaseNodeOps):
    pass


@make_binary(' + ')
class Add(BaseNodeOps):
    pass


@make_binary(' - ')
class Sub(BaseNodeOps):
    pass


@make_binary(' / ')
class Divide(BaseNodeOps):
    pass


@make_binary(' * ')
class Multiply(BaseNodeOps):
    pass


@make_binary(' % ')
class Mod(BaseNodeOps):
    pass


@make_binary(' > ')
class Greater(BaseNodeOps):
    pass


@make_binary(' >= ')
class GreaterEqual(BaseNodeOps):
    pass


@make_binary(' < ')
class Less(BaseNodeOps):
    pass


@make_binary(' <= ')
class LessEqual(BaseNodeOps):
    pass


#
# Variables
#



@name_args(['variable', 'index'])
class ArrayElement(BaseNodeOps):
    @property
    def x(self):
        return Component(str(self), 'x')
    @property
    def y(self):
        return Component(str(self), 'y')
    def address(self):
        return Address(str(self))
    def __str__(self) -> str:
        return str(self.variable) + '[' + str(self.index) + ']'


@name_args(['name', 'type', 'size', 'array'])
class Variable(BaseNodeOps):
    @property
    def x(self):
        return Component(self.name, 'x')
    @property
    def y(self):
        return Component(self.name, 'y')
    def address(self):
        return Address(self.name)
    def declaration(self):
        if self.size is not None:
            return Declaration(f'{self.type} {self.name}[{self.size}];')
        return Declaration(f'{self.type} {self.name};')
    def argument(self):
        if self.array:
            return f'{self.type} *{self.name}'
        return f'{self.type} {self.name}'
    def __str__(self) -> str:
        return str(self.name)
    def __getitem__(self, idx):
        return ArrayElement(self.name, idx)


class ComplexLiteral(BaseNodeOps):
    def __str__(self):
        return '{' + str(self.args[0]) + ', ' + str(self.args[1]) + '}'


class Group(BaseNodeOps):
    def __str__(self):
        return '(' + str(self.args[0]) + ')'

B = Group


#
# Control flow
#

class Block(BaseNode):
    def __str__(self):
        return '{' + join('\n', self.args) + '}'

@name_args(['condition', 'body'])
class If(BaseNode):
    def __str__(self) -> str:
        f = 'if( ' + str(self.condition) + ')'
        f += '{' + join('\n', self.body) + '}'
        return f


#
# Functions
#

@name_args(['name', 'arguments', 'templates', 'qualifier', 'body'])
class Function(BaseNode):
    def __str__(self) -> str:
        f = ''
        if self.templates:
            f += 'template<' + str(self.templates) + '>'
        if self.qualifier is not None:
            f += self.qualifier + ' '
        f += ' void ' + self.name
        f += '(' + str(self.arguments) + ')'
        f += '{' + join('\n', self.body) + '}'
        return f

@name_args(['name', 'arguments', 'templates', 'launch_params'])
class Call(BaseNode):
    def __str__(self) -> str:
        f = self.name
        if self.templates:
            f += '<' + join(', ', self.templates) + '>'
        if self.launch_params:
            f += '<<<' + join(',  ', self.launch_params) + '>>>'
        f += '(' + join(', ', self.arguments) + ');'
        return f

#
# Misc
#

return_statement = String("return;")
line_break = String("\n\n")
sync_threads = String("__syncthreads();")
