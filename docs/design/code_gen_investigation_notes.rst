=========================================
FFT code generator investigation notes
=========================================

.. sectnum::

.. contents:: Table of contents

genfft - FFTW codlete generator
~~~~~~~~~~~~~~~~~~~~~~~~~

genfft is the engine to generate "codelets" in FFTW. It is written in Objective Caml. genfft covers Cooley-Tukey equation, prime-factor, split-radix and Rader algorithms. genfft has 4 phases: creation, simplicfication, scheduling, target output. During creation, genfft creats codelet as a node of directed acyclic graph(DAG).

- Further reading:

  - http://supertech.csail.mit.edu/papers/pldi99.pdf
  - http://www.fftw.org/fftw-paper-ieee.pdf
  
- Quick try:

  - apt install ocamlbuild
  - apt install ocaml-nox
  - git clone git@github.com:FFTW/fftw3.git
  - cd genfft
  - ocamlbuild -classic-display -libs unix,nums gen_hc2c.native
  - ./gen_hc2c.native -n 8 > hc2c_8.cpp
  


SPIRAL
~~~~~~~~~~~~~~~~~~~~

SPIRAL is written in SPL(an acronym from Signal Processing Language). It is designed to target on automatic code gereration for various hardware platforms(CPU/DSP/FPGA). The architecture of SPIRAL on high levels are:

 - Algorithm level
 - Implementation level
 - Evaluation level

SPIRAL searches at compile-time over a space of mathematically equivalent
formulas expressed in SPL.

- Further reading:

  - https://users.ece.cmu.edu/~franzf/papers/si-spiral.pdf
  - http://spiral.ece.cmu.edu:8080/pub-spiral/pubfile/dftcomp_96.pdf
  
- Quick try:

  - Requires CMake >= 3.14
  
    - sudo apt remove --purge cmake
    - hash -r
    - sudo snap install cmake --classic

  - git clone git@github.com:spiral-software/spiral-software.git
  - mkdir build
  - cd build
  - cmake ..
  - make install
  - make test
  - ctest
  

Open discussion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Do we need DAG?
- How does SPIRAL optimize for NV platform?
