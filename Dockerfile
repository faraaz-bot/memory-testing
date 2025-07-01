
FROM ubuntu:24.04

# Build-time credentials for the private SCALE repo --
ARG CUSTOMER_NAME
ARG CUSTOMER_PASSWORD

# Core system utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl ca-certificates gnupg apt-transport-https lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install ROCm 6.3.1 user-land (no DKMS) 
RUN wget https://repo.radeon.com/amdgpu-install/6.3.1/ubuntu/noble/amdgpu-install_6.3.60301-1_all.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends ./amdgpu-install_6.3.60301-1_all.deb && \
    rm amdgpu-install_6.3.60301-1_all.deb && \
    #
    # `amdgpu-install` meta-script: user-land only
    #
    amdgpu-install -y \
        --usecase=rocm          \
        --rocmrelease=6.3.1     \
        --no-dkms               \
        --no-32                 \
        --accept-eula=yes && \
    #
    # Add root to GPU access groups
    groupadd -f render && usermod -aG render,video root && \
    #
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/rocm/bin:${PATH}"

#  Dev tools for building the project 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git python3-setuptools python3-wheel \
    && rm -rf /var/lib/apt/lists/*

#  Authenticate & enable SCALE unstable repo 
RUN printf 'machine unstable-nonfree-pkgs.scale-lang.com\n\
login %s\n\
password %s\n' "$CUSTOMER_NAME" "$CUSTOMER_PASSWORD" \
    > /etc/apt/auth.conf.d/scale.conf && \
    chmod 600 /etc/apt/auth.conf.d/scale.conf && \
    #
    wget --quiet --http-user="$CUSTOMER_NAME" \
                 --http-password="$CUSTOMER_PASSWORD" \
        https://unstable-nonfree-pkgs.scale-lang.com/${CUSTOMER_NAME}/deb/dists/noble/main/binary-all/scale-repos.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends ./scale-repos.deb && \
    rm scale-repos.deb

# Install SCALE enterprise (unstable) 
RUN apt-get update && \
    apt-get install -y --no-install-recommends scale-unstable && \
    rm -rf /var/lib/apt/lists/*

# Expose SCALE environment variables
ENV SCALE_DIR=/opt/scale
ENV PATH=${SCALE_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${SCALE_DIR}/lib:${LD_LIBRARY_PATH}

# Workspace 
WORKDIR /workspace

# Copy your project files (adjust as needed)
COPY CMakeLists.txt compile.sh *.cu /workspace/

RUN chmod +x /workspace/compile.sh

# Default entrypoint 
CMD ["bash"]
