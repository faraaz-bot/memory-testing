Code generator
==============

Rationale
---------

The previous code generator:

* is not (well) documented
* is based on string concatenation
* can be (subjectively) cumbersome to work with

Ideadlly, a new code generator:

* would be concise
* resonably documented
* based on an abstract-syntax-tree (AST)
* easy to work with

ASTs allow generated code to be transformed and manipulated before
being emitted.  A concrete example of this for FFT kernels would be:
automatically translating a kernel from interleaved to planar format.

How the generator is designed and implemented is crucial for both
conciseness and ease-of-use.


Design considerations
---------------------

Some ideas from the team:

* should we treat "batch == 4th dimension"
* should we start to design to support N dimension?
* how to organize all device and global function in files?
* how to make it easy to debug/backtrack/hack the generator and generated code?
* how easy to insert asm code, and/or maintain arch specific code?
* auto-tuning (at least partial) capability or interfaces
* support generation of HIP or MLIR

Related projects:

* Spiral FFT
* FFTX
* FFTW

Ideas gleaned from looking at related projects:

* FFTW: GURU interface
* FFTX: codelets?


Known issues
------------

The current generator uses the single-precision large-threshold when
generating double-precision kernels.


Required kernels (scope)
------------------------

For rocFFT, we need/want to generate:

* Host functions to launch the FFT kernels
* Tiled (row/column) + strided + batched Stockham kernels for arbitrary factorisations

Kernels need to handle all combinations of:

* single/double precision
* in-place/out-of-place
* planar/interleaved
* real/complex
* small/large twiddle tables
* unit/non-unit stride

Ideally any configuration/runtime parameters required by the kernels
would be defined in a single place to avoid repetition between rocFFT
and the generator.

We have flexibility in handling these combinations at compile-time or
run-time.  For example, multiple kernels could be generated for
single/double precision, but unit/non-unit stride could be handled at
runtime.

Tiling
^^^^^^

XXX

Strides
^^^^^^^

XXX

Batches
^^^^^^^

XXX

Large twiddle tables
^^^^^^^^^^^^^^^^^^^^

XXX

Launching
^^^^^^^^^

Currently kernels are lauched with:

* dimension
* number of blocks (batches)
* number of threads (threads per batch; kernel parameter)
* stream
* twiddle table
* length(s)
* strides
* batch count
* in/out buffers

We have a lot of flexibility here.


Implementation
--------------

The code generator will by implemented in Python; targetting version
3.6 and using only standard modules.

The AST will be represented as a tree structure, with nodes in the
tree representing operations, such as assignment, addition, or a block
containing multiple operations.  Nodes will be represented as objects
(eg, `Add`) extending the base class `BaseNode`.  Operands will be
stored in a simple list called `args`:

.. code_block: python

    class BaseNode:
        args: List[Any]


To facilitate building ASTs, the base node will have a constructor
that simply stores it's arguments as operands:

.. code_block: python

    class BaseNode:
        args: List[Any]
        def __init__(self, *args, **kwargs):
            self.args = list(args)


To facilitate rewriting ASTs, node object's constructors should accept
a simple list of argument/operands.

This, for example, allows a depth-first tree re-write to be
implemented trivially as:

.. code_block: python

    def depth_first(x, f):
        '''Depth first traveral of the AST in 'x'.  Each node is transformed by 'f(x)'.'''
        if isinstance(x, BaseNode):
            y = type(x)(*[ depth_first(a, f) for a in x.args ])
            return f(y)
        return f(x)

To emit code, each node must implement `__str__`.  For example:

.. code_block: python

    class Add(BaseNode):
        def __str__(self):
            return ' + '.join([ str(x) for x in self.args ])
