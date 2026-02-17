# Installs NVSHMEM and DeepEP.

echo "Installing DeepEP ${PRETRAIN_DEEPEP_VERSION}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

# Install NVSHMEM (required by DeepEP)
echo "Installing NVSHMEM ${PRETRAIN_NVSHMEM_VERSION}"
mkdir -p nvshmem
pushd nvshmem
wget https://developer.download.nvidia.com/compute/redist/nvshmem/${PRETRAIN_NVSHMEM_VERSION}/source/nvshmem_src_cuda12-all-all-${PRETRAIN_NVSHMEM_VERSION}.tar.gz
tar -xf nvshmem_src_cuda12-all-all-${PRETRAIN_NVSHMEM_VERSION}.tar.gz
pushd nvshmem_src

python3 -m pip install cmake
GDRCOPY_HOME=$(echo $CPATH | tr ':' '\n' | grep gdrcopy | head -1 | sed 's|/include||')

NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
GDRCOPY_HOME=${GDRCOPY_HOME} \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/nvshmem -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build/ -j$(nproc) --target install

popd  # nvshmem_src
popd  # nvshmem

export NVSHMEM_DIR=${TARGET_DIR}/nvshmem
export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}
export PATH=${NVSHMEM_DIR}/bin:${PATH}

# Download DeepEP and build
git clone https://github.com/deepseek-ai/DeepEP
pushd DeepEP
git checkout ${PRETRAIN_DEEPEP_VERSION}
sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh
python -m pip install --no-build-isolation .

popd  # DeepEP

popd  # ${TARGET_DIR}/src
deactivate
