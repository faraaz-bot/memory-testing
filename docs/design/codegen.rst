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

Launching FFT kernels
---------------------

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
