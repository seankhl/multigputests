
#include <array>

#include <cuda_runtime_api.h>
#include <cuda.h>

__device__ float dist(const float, const float, const float, const float);

__global__ void timestep(const int atoms_sz, const float threshold, 
                         const float4 *vals_old, float4 *vals_new,
                         const float4 *ghost_lo, const int ghost_lo_sz,
                         const float4 *ghost_hi, const int ghost_hi_sz);

