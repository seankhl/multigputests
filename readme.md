
What This Is
============

This is a toy simulation code for testing different ways of achieving multi GPU
parallelism. There is a GPUDirect version (native CUDA) and an MPI version
(CUDA-aware MPI). Feasibly we could look at solutions that incorporate
thrust/nccl in the future. We could also look at a native CUDA version that
uses remote memory access instead of GPU<->GPU communications.

To run the mpi code, execute the following command:

```bash
/usr/local/mpi-cuda/bin/mpirun -np 1 --mca btl openib,self mpi_test 
```

Helpful Links
=============

Link to version of Open MPI used:
  * https://github.com/open-mpi/ompi-release/tree/v1.10

How to build Open MPI with CUDA-Aware support:
  * https://www.open-mpi.org/faq/?category=building#build-cuda
  * https://www.open-mpi.org/faq/?category=runcuda

NVIDIA docs about CUDA-Aware MPI:
  * https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/
  * https://devblogs.nvidia.com/parallelforall/benchmarking-cuda-aware-mpi/

Other helpful/interesting links:
  * https://community.mellanox.com/community/support/software-drivers/rdma-software-for-gpu/blog/2014/01/27/using-gpudirect-rdma-with-mpi

Links to info about multi GPU programming:
  * http://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf
  * http://www.math.wsu.edu/math/kcooper/CUDA/13CUDAblock.pdf

Helpful CUDA wrappers for future reference:
  * https://github.com/thrust/thrust
  * https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/
  * https://github.com/NVIDIA/nccl

How to run CUDA samples (there is a multi GPU sample):
  * http://docs.nvidia.com/cuda/cuda-samples/index.html#getting-cuda-samples
