Tuning Framework Design Document for rocFFT
=========================================

Copyright and Disclaimer
---------

DISCLAIMER

The information contained herein is for informational purposes only,
and is subject to change without notice. While every precaution has
been taken in the preparation of this document, it may contain
technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information.
Advanced Micro Devices, Inc. makes no representations or warranties
with respect to the accuracy or completeness of the contents of this
document, and assumes no liability of any kind, including the implied
warranties of noninfringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein.  No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document.  Terms and limitations applicable
to the purchase or use of AMD’s products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and
Conditions of Sale.

AMD is a trademark of Advanced Micro Devices, Inc.  Other product
names used in this publication are for identification purposes only
and may be trademarks of their respective companies.

Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.


Target
--------

* Provide architecture specific optimized solution for given FFT problem.
* Reduce manual tunning as many as possible.
* Provide most simple optimal kernel code with RTC.
* Heuristic tuning based on known knowledge to cover/support potential unknown cases.

Proposal
---------

Overall design
^^^^^^^^^^^^^^^^^^^

2 level hierarchy tuning: FFT plan decomposition tuner and FFT kernel tuner.
Given any FFT problem on user level, rocFFT decomposes

FFT plan decomposition tuner(PlanTuner)
^^^^^^^^^^^^^^^^^^^
Assuming KernelTuner provides optimal solution on low kernel level, PlanTuner should find 
optimal solutions to decompose a given FFT problem from user level(root FFT tree node) to 
a sequential FFT kernels(leaf FFT tree nodes).

* PlanTuner searches and stores solution per architecture for give problem.
* A preliminary search of recursive decomposition might go for DFS or BFS.
* Solution searching might be classified into: algorithm selection, layout selection, and
  1D large decomposition layout selection.
* Guided rules for searching might include: less kernel numbers.
* Reuse exsiting sub-plans.
* Key GPU architecture spec might be considered as inputs, such as LDS size.


Solution storge
~~~~
The solution will store with recursive solution map. 



FFT kernel tuner(KernelTuner)
^^^^^^^^^^^^^^^^^^^