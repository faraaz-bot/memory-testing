Example for how to query other kinds of device properties such as cache sizes using HSA API.
ROCr Runtime GitHub page and their docs have more details, such as other enums that can be passed to
`hsa_agent_get_info()`

Following command can be used to build, assuming amdclang++ or other appropriate compiler is added to PATH:
`cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -DCMAKE_CXX_COMPILER=amdclang++ ..`
