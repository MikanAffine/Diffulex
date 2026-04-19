#!/bin/bash
git submodule update --init --recursive

cd thirdparty/DeepEP
rm -rf build deep_ep.egg-info
find . -name "*.so" -delete
find . -name "*.o" -delete

unset NVSHMEM_DIR
unset LD_LIBRARY_PATH

export DISABLE_NVSHMEM=1
export DISABLE_SM90_FEATURES=1
export TORCH_CUDA_ARCH_LIST="8.6"
export DISABLE_AGGRESSIVE_PTX_INSTRS=1

python setup.py install