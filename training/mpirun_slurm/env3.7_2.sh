#!/bin/bash

source ~/torch/venv_torch3.7/bin/activate
module unload compiler/rocm/dtk-22.10.1
module unload mpi/hpcx/2.11.0/gcc-7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
#module load compiler/intel/intel-compiler-2020.1.217

export ROCM_PATH=/public/home/yuguo960516yuguo/dtk/dtk-23.04
export ROCM_SOURCE_DIR=${ROCM_PATH}
echo $ROCM_PATH
export HIP_PATH=${ROCM_PATH}/hip
export AMDGPU_TARGETS="gfx926;gfx906"
export PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:$PATH

export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/opencl/lib/x86_64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${ROCM_PATH}/include:${ROCM_PATH}/hip/include/hip:${ROCM_PATH}/llvm/include:/opencl/include:${ROCM_PATH}/include/rocrand:${ROCM_PATH}/include/hiprand
export CPLUS_INCLUDE_PATH=${ROCM_PATH}/include:${ROCM_PATH}/hip/include/hip:${ROCM_PATH}/llvm/include:/opencl/include:${ROCM_PATH}/include/rocrand:${ROCM_PATH}/include/hiprand
export PATH=${ROCM_PATH}/miopen/bin:${ROCM_PATH}/rocblas/bin:${ROCM_PATH}/hipsparse/bin:$PATH
export LD_LIBRARY_PATH=${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:$LD_LIBRARY_PATH
export MIOPEN_SYSTEM_DB_PATH=${ROCM_PATH}/miopen/share/miopen/db/
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH

export RCCL_PATH=$ROCM_PATH/rccl
export NCCL_PATH=$ROCM_PATH/rccl
export LD_LIBRARY_PATH=$RCCL_PATH/lib:$LD_LIBRARY_PATH

export PYTHON_VENV_PATH=/public/home/yuguo960516yuguo/torch/venv_torch3.7
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/Python3.7d/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/rccl-net_lib/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/ucx_lib/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/lmdb_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/opencv_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/gflags_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/glog_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/package/openblas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/torch/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/yuguo960516yuguo/torch/venv_torch3.7/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH
export PATH=/public/home/yuguo960516yuguo/torch/openssl/bin:$PATH

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_COMPILE_PARALLEL_LEVEL=1
export NCCL_PLUGIN_P2P=ucx
#export NCCL_IB_HCA=mlx5_0
export NCCL_IB_HCA=mlx5
export RCCL_NCHANNELS=2
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=SYS
#export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_LEVEL=5

#export ROCBLAS_LAYER=3

