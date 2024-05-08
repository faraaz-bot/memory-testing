
FFT plan optimazation logic
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



--------

The below design doesn't cover multiple nodes/multiple devices.


* Pre/intermediate/post processing fusion

Any pre/intermediate/post processing with FFT can be fused? (callback/spectral ops/convolution use case)
If yes, check the available path/kernels to fuse.

* FFT algrithms selection
    * R2C/C2R
          * Embedded with C2C
          * Separate pass
    * C2C
          * Stockham for supported radixes
          * Bluestein for prime

* Memory layout algrithmic selection

  Apply batched FFT on each dimension recrucevily. The key point is to choose
  the proper bandle FFTs for a workgroup to work with.
    * Multiple dimensions FFT decomposition
      For a 3D FFT, there are 3 choices:(1) SBRR/SBCC/SBCC, (2) SBRC/SBRC/SBRC, (3) SBCR/SBCR/SBCR
    * Single dimension large FFT decomposition would have Cooley-Turkey decomposition involved.
      LDS capicity is the key to detemine how many rounds will be decomposed.
      The more pseudo 2D decomposition close to square, the better.

* Kernel selection order
  Fit into single kernel ---> Fit into patial pass ---> large FFT decomposition

* Buffer assigment/adjustment
    * We would try first and assume in-place buffer assigment unless SBRC/SBCR/Transpose require out-placement.
    * For large pow-of-2 SBRC/SBCR/Transpose, we might need diangial transpose or extral padding to avoid L2 channel hashing conflict.

* Cache hierarchy optimazation
    * For multiple dimensions FFT won't fit into MALL(>=MI300), batched sliced FFTs might help improve MALL hit rate.


* Remarks:

    * Typically, we don't think too much on leaf nodes fusion from sibling trees.
    * No optimazation for special strided FFTs unless important customers ask.
    * All inter-nodes are in interleave format even for plannar FFTs.
