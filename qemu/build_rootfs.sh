# Install pre-reqs
sudo apt install debootstrap qemu-system-misc qemu-user-static binfmt-support dpkg-cross --no-install-recommends

# Generate minimal bootstrap rootfs (temp-rootfs is a folder that must exist)
sudo debootstrap --arch=riscv64 --foreign noble ./temp-rootfs http://ports.ubuntu.com/ubuntu-ports

sudo mkdir -p ./temp-rootfs/etc

# chroot to it and finish debootstrap
sudo chroot temp-rootfs /bin/bash

/debootstrap/debootstrap --second-stage

# Add package sources
cat >/etc/apt/sources.list <<EOF
deb http://ports.ubuntu.com/ubuntu-ports noble main restricted

deb http://ports.ubuntu.com/ubuntu-ports noble-updates main restricted

deb http://ports.ubuntu.com/ubuntu-ports noble universe
deb http://ports.ubuntu.com/ubuntu-ports noble-updates universe

deb http://ports.ubuntu.com/ubuntu-ports noble multiverse
deb http://ports.ubuntu.com/ubuntu-ports noble-updates multiverse

deb http://ports.ubuntu.com/ubuntu-ports noble-backports main restricted universe multiverse

deb http://ports.ubuntu.com/ubuntu-ports noble-security main restricted
deb http://ports.ubuntu.com/ubuntu-ports noble-security universe
deb http://ports.ubuntu.com/ubuntu-ports noble-security multiverse
EOF

# Install essential packages
apt-get update
apt-get install --no-install-recommends -y util-linux haveged openssh-server systemd kmod initramfs-tools conntrack ebtables ethtool iproute2 iptables mount socat ifupdown iputils-ping vim dhcpcd5 neofetch sudo chrony gcc make g++ openssl libssl-dev autoconf m4 flex bison pkg-config python3.12-dev git nano wget pip python3-venv aptitude automake rustup gedit file btop cmake libjpeg-dev gfortran libffi-dev libpq-dev python3-dev

rustup default stable

# From here, install Pytorch from source as usual.

# Create virtual enviroment
python -m venv pythonField
source pythonField/bin/activate

# Clone and move to version
git clone https://github.com/pytorch/pytorch --recursive && cd pytorch
# git checkout v2.9.1 # optional, you can build master if you are brave Hacer con 2.11, si no no funciona (no entiendo pq, torchvision 0.24 debería ir con torch 2.9???)
git checkout 3854d691ce36d00f5acbebc03c1f53f0f1e00d9b
git submodule update --init --recursive

# Change wrong line
sed -i '26s/.   */STRING(REGEX REPLACE "^.*(sve).*$" "\\1" SVE_THERE "${CPUINFO}")/' cmake/Modules/FindARM.cmake

# Install requeriments
pip install -r requeriments.txt

# Ninja fails, so uninstall
pip uninstall ninja

cd ../

# Create script to create wheel
cat <<EOF >mySetup.sh
export NO_CUDA=1
export NO_DISTRIBUTED=1
export NO_MKLDNN=1
export BUILD_TEST=0 # for faster builds
export MAX_JOBS=4 # I have 8 cores
export NO_NNPACK=1 # update July 19, this is optional, can build with NNPACK
export NO_QNNPACK=1 # same as above, can be omitted
python setup.py bdist_wheel
EOF

# Execute the script
sh mySetup.sh

# Install wheel
cd dist
pip install torch-2.11.0a0+git3854d69-cp312-cp312-linux_riscv64.whl

# Move to banana
scp torch-2.11.0a0+git3854d69-cp312-cp312-linux_riscv64.whl acorroch_local@banana:/home/acorroch_local

# Now we need to install torch vision

cd ../..

git clone https://github.com/pytorch/vision.git --recursive && cd vision
git checkout release/0.24

python setup.py bdist_wheel 
cd dist/
pip install torchvision-0.25.0a0+1e53952-cp312-cp312-linux_riscv64.whl

scp torchvision-0.24.1+d801a34-cp312-cp312-linux_riscv64.whl acorroch_local@banana:/home/acorroch_local

# BLIS
git clone https://github.com/flame/blis.git
cd blis
./configure -p /opt/blis rv64i # auto buscar generic y paralel rv64iv sifive_rvv sifive_x280 on bachs crecientes seleccionar modelos retantes en rendimiento 
make -j4
make install

sudo scp -r blis acorroch_local@banana:/home/acorroch_local/

# usarlo en pytorch
# export BLIS_HOME="/home/acorroch_local/blisVect/blisVect"
# export BLIS_HOME="/home/acorroch_local/blisGeneric/blisGeneric"
export BLIS_HOME="/opt/blis"
export CMAKE_PREFIX_PATH="${BLIS_HOME}:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${BLIS_HOME}/lib:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${BLIS_HOME}/lib:${DYLD_LIBRARY_PATH}"

cd ../pytorch

pip install scikit-build

cat <<EOF >myBlisSetup.sh 
export BLIS_HOME="/opt/blis"
export CMAKE_PREFIX_PATH="${BLIS_HOME}:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${BLIS_HOME}/lib:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${BLIS_HOME}/lib:${DYLD_LIBRARY_PATH}"
export USE_BLAS=1
export USE_BLIS=1
export USE_OPENBLAS=0
export NO_CUDA=1
export NO_DISTRIBUTED=1
export NO_MKLDNN=1
export BUILD_TEST=0 # for faster builds
export MAX_JOBS=4 # I have 8 cores
export NO_NNPACK=1 # update July 19, this is optional, can build with NNPACK
export NO_QNNPACK=1 # same as above, can be omitted
python setup.py bdist_wheel
EOF

sh myBlisSetup.sh
#pip install --no-build-isolation -v -e .
