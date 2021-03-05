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

* Should we treat "batch == 4th dimension"?
* Should we start to design to support N dimension?
* How to organize all device and global function in files?
* How to make it easy to debug/backtrack/hack the generator and generated code?
* How easy to insert asm code, and/or maintain arch specific code?
* Auto-tuning (at least partial) capability or interfaces
* Support generation of HIP or MLIR

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

Fundamentally, all multidimensional and batched FFTs can be written in
terms of 1D transforms (with affine indexing).  As such, an FFT is
broken down into:

* A *host* function that is aware of dimensions, strides, batches, and
  tiling.  This function would be responsible for determining how the
  problem will be broken down into GPU thread blocks.
* A *global* function that is aware of GPU thread blocks, dimensions,
  strides, batches, and tiling.  This function would be responsible
  for determining offsets and strides for the device function, and
  declaring LDS memory buffers.
* A *device* function that is passed offsets and strides, and is aware
  of GPU threads.  The device function would perform a (short) 1D
  transform.

A device function may be called so that a thread block is actually
transforming multiple batches.  As such, indexes (the spatial index in
the FFT) should be computed as:

.. code-block::

   int fft_index = threadIdx.x % length;


Tiling
^^^^^^

Launching device kernels in a way that traverses memory in tiles will
be handled at the host/global level.

XXX large twiddle tables?

Strides and batches
^^^^^^^^^^^^^^^^^^^

Host
~~~~

Host/global functions should support arbitrary dimensions, lengths,
strides, offsets, and batches.

Users should be allowed to store their arrays arbitrarily.  For an
:math:`N` dimensional dataset, the array index :math:`a` corresponding
to indices :math:`(i_1,\ldots,i_N,i_b)`, where :math:`i_b` is the
batch index, is given by

.. math::

   a(i_1,\ldots,i_N,i_b) = \sum_{d=1}^N s_d i_d + s_b i_b

where :math:`s_d` is the stride along dimension :math:`d`.  To support
these strides, the device function to compute the FFT along dimension
:math:`D` would be passed:

.. code-block:: c

   int offset = 0;
   offset += batch_index * batch_stride;
   for (int d=0; d < N; ++d)
     if (d != D)
       offset += spatial_index[d] * strides[d];

   int stride = strides[D];

For example, in three dimensions, to compute the FFT along the
y-dimension given x and z indicies ``i`` and ``k`` for batch ``b``,
the device function would be passed:

.. code-block:: c

   int offset = 0;
   offset += b * batch_stride;
   offset += i * strides[0];
   offset += k * strides[2];

   int stride = strides[1];

Device
~~~~~~

Device functions should support arbitrary offsets and strides.  Array
indexes in device functions should be computed as:

.. code-block::

   int array_index = offset + fft_index * stride;


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
(eg, ``Add``) extending the base class ``BaseNode``.  Operands will be
stored in a simple list called ``args``:

.. code-block:: python

    class BaseNode:
        args: List[Any]


To facilitate building ASTs, the base node will have a constructor
that simply stores it's arguments as operands:

.. code-block:: python

    class BaseNode:
        args: List[Any]
        def __init__(self, *args, **kwargs):
            self.args = list(args)


To facilitate rewriting ASTs, node object's constructors should accept
a simple list of argument/operands.

This, for example, allows a depth-first tree re-write to be
implemented trivially as:

.. code-block:: python

    def depth_first(x, f):
        '''Depth first traveral of the AST in 'x'.  Each node is transformed by 'f(x)'.'''
        if isinstance(x, BaseNode):
            y = type(x)(*[ depth_first(a, f) for a in x.args ])
            return f(y)
        return f(x)

To emit code, each node must implement ``__str__``.  For example:

.. code-block:: python

    class Add(BaseNode):
        def __str__(self):
            return ' + '.join([ str(x) for x in self.args ])


Launching
^^^^^^^^^
