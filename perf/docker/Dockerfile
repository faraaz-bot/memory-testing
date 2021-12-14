FROM compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang:9237-ubuntu-20.04-cuda-11.x-stg1
RUN apt update && apt install -y cmake git rocm-cmake libfftw3-dev libboost-program-options-dev libgtest-dev
RUN apt install -y python3-dev python3-click python3-numpy python3-sympy python3-scipy python3-pandas python3-pip patchelf 
RUN pip3 install plotly

ENV HIPCC_COMPILE_FLAGS_APPEND= HIPCC_LINK_FLAGS_APPEND= GIT_SSH_COMMAND='ssh -i /x/.ssh/id_rsa -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no'
