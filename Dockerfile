# Base image
FROM ubuntu:24.04

# Build arguments
ARG CUSTOMER_NAME
ARG CUSTOMER_PASSWORD

# Dependencies to pull from https
RUN apt update && \
    apt install -y --no-install-recommends \
        wget \
        apt-transport-https \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Scale requires ROCm 6.3.1 and amdgpu 
RUN wget https://repo.radeon.com/amdgpu-install/6.3.1/ubuntu/noble/amdgpu-install_6.3.60301-1_all.deb && \
    apt update && \
    apt install -y --no-install-recommends ./amdgpu-install_6.3.60301-1_all.deb && \
    rm amdgpu-install_6.3.60301-1_all.deb && \
    apt update && \
    apt install -y --no-install-recommends python3-setuptools python3-wheel && \
    groupadd -f render && usermod -aG render,video root && \
    apt update && \
    apt install -y --no-install-recommends "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" && \
    apt install -y --no-install-recommends amdgpu-dkms && \
    apt update

ENV PATH="/opt/rocm/bin:${PATH}"

RUN apt update && \
    apt install -y --no-install-recommends \
      cmake \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tell apt to authenticate to the repo
RUN printf 'machine unstable-nonfree-pkgs.scale-lang.com\n\
login %s\n\
password %s\n' "$CUSTOMER_NAME" "$CUSTOMER_PASSWORD" \
	> /etc/apt/auth.conf.d/scale.conf \
	&& chmod 700 /etc/apt/auth.conf.d/scale.conf

# Finish authentication and install deb repos
RUN wget --http-user="$CUSTOMER_NAME" \
         --http-password="$CUSTOMER_PASSWORD" \
         https://unstable-nonfree-pkgs.scale-lang.com/$CUSTOMER_NAME/deb/dists/noble/main/binary-all/scale-repos.deb && \
    apt update && \
    apt install -y --no-install-recommends ./scale-repos.deb \
    && rm scale-repos.deb

# Install SCALE enterprise
RUN apt update && \
    apt install -y --no-install-recommends scale-unstable \
    && rm -rf /var/lib/apt/lists/*

# Set scale path variables
ENV SCALE_DIR=/opt/scale
ENV PATH=${SCALE_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${SCALE_DIR}/lib:${LD_LIBRARY_PATH}

# Add user to the `video` group
RUN groupadd -f video && usermod -aG video root

# Add working directory
WORKDIR /workspace/

# Layer in scale tests to workspace
COPY CMakeLists.txt compile.sh *.cu /workspace/

RUN chmod +x /workspace/compile.sh 

CMD ["bash"]
