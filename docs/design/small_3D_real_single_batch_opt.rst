

Target
^^^^^^^^^^^^^^^^^^^^^
To speed up 3D R2C/C2R single batch small cases.

The original requirements are from AMBER for case 64^3, R2C/C2R, single precision, outplace. But it should benefit other projects like GROMACS, NAMD as well.

Proposal
^^^^^^^^^^^^^^^^^^^^^

(1) Reduce global memory access as many as possible

There is possibility to reduce 3 times global memory access down to 2 or 1, which depends on the problem size, lds size, and L2 cache size.
First, check Malcolm's `proposal. <https://teams.microsoft.com/l/file/9C796761-79E1-4F7B-9776-353EAB1E634F?tenantId=3dd8961f-e488-4e60-8e11-a82d994e183d&fileType=pdf&objectUrl=https%3A%2F%2Famdcloud.sharepoint.com%2Fsites%2FMLSELibrariesDevelopment%2FShared%20Documents%2FrocFFT%2Frocfft_runtime_opts.pdf&baseUrl=https%3A%2F%2Famdcloud.sharepoint.com%2Fsites%2FMLSELibrariesDevelopment&serviceName=teams&threadId=19:98a3fc6f793f4ebc95a31990f3e20da0@thread.skype&groupId=ac2278ba-8e20-4f39-b829-5ad102f224ef>`_

And for 64^3 R2C/C2R, if it has SBCC/SBCC/SBRR, we also have choice to choose to fuse which 2 of them. 

(2) Reduce kernel dispatches

AMBER complained kernel launch overhead when rocFFT applying 5-6 kernels, even down to 3.

For a pure copy/transpose of float2 108x216x216 on MI100, rocprof shows hipLaunchKernel avgNs is 16686, while the kernel itself takes 85535.

For any multiple kernels scenario, it might be worth dispatching them only once if possible.

The hip cooperative groups seems having too much overhead. Persistent kernel with producer-consumer control might help.

It potentially would help prime sizes with Bluestein algorithm. 


(3) Other general optimizations for 1D FFTs are not mentioned here.

Reference
^^^^^^^^^^^^^^^^^^^^^
- A previous discussion `SWEDV-204997.pdf <https://teams.microsoft.com/l/file/CFDCC528-BDF7-4B73-8EFF-91327AEBA401?tenantId=3dd8961f-e488-4e60-8e11-a82d994e183d&fileType=pdf&objectUrl=https%3A%2F%2Famdcloud.sharepoint.com%2Fsites%2FMLSELibrariesDevelopment%2FShared%20Documents%2FrocFFT%2FSWDEV-204997.pdf&baseUrl=https%3A%2F%2Famdcloud.sharepoint.com%2Fsites%2FMLSELibrariesDevelopment&serviceName=teams&threadId=19:98a3fc6f793f4ebc95a31990f3e20da0@thread.skype&groupId=ac2278ba-8e20-4f39-b829-5ad102f224ef>`_
