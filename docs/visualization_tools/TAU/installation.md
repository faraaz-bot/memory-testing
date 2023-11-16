
# Using Tau on Lockhart

## 1. Download e4s docker
    Go to e4s.io and select a version
    wget https://oaciss.uoregon.edu/e4s/images/23.02/e4s-rocm-x86_64-23.02.sif

    apt-get install -y \
      build-essential \
      libseccomp-dev \
      pkg-config \
      squashfs-tools \
      cryptsetup \
      libglib2.0-dev

## 2. Get GO
    sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.20.4.linux-amd64.tar.gz
    export PATH=$PATH:/usr/local/go/bin
    go version
      
## 3. Get singularity
    export VERSION=3.11.3 # adjust this as necessary
    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz
    tar -xzf singularity-ce-${VERSION}.tar.gz
    
## 4. Build singularity    
    cd singularity-ce-${VERSION}
    ./mconfig
    make -j4 -C builddir
    sudo make -j4 -C builddir install

## 5. Run the e4s image 
    singularity run --rocm ./e4s-rocm-x86_64-23.02.sif

## 6. Test Tau

    Singularity> module av tau

    --------------------------------------- 
    /spack/share/spack/lmod/linux-ubuntu20.04-x86_64/mpich/4.1-v4dkt2i/Core 
    ----------------------------------------
    tau/2.32-rocm    tau/2.32 (D)

    mpirun --bind-to core -n 2 tau_exec -ebs ./mpi_benchmark ALLREDUCE 16777216 20




# Tau on Frontier - Cray compiler

## 1. Setup
    module purge
    source /sw/frontier/ums/ums002/E4S/23.05/PrgEnv-gnu/module-use.sh
    module load tau/2.32-rocm
    module unload darshan-runtime
    module load cray-mpich

## 2. Verify modules

    Currently Loaded Modules:
    1) craype/2.7.19        4) craype-network-ofi      7) gcc/11.2.0       10) rocm/5.3.0         13) papi/6.0.0.1
    2) cray-dsmml/0.2.2     5) cray-libsci/22.12.1.1   8) hsi/default      11) cray-mpich/8.1.23  14) pdt/3.25.1
    3) libfabric/1.15.2.0   6) PrgEnv-gnu/8.3.3        9) DefApps/default  12) libunwind/1.6.2    15) tau/2.32-rocm


## 3. Test
    export TAU_TRACE=1; export TAU_TRACE_FORMAT=otf2 
    tau_exec -rocm -ebs   ./fft_performance.x -nx 512 -ny 512 -nz 512

## 4. Create pack for visualization
    paraprof --pack fft_test.ppk 

## 5. Open in TAU GUI
    paraprof fft_test.ppk

This can also be oppened in Perffeto.
