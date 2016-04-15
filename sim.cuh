
#include <cuda_runtime_api.h>
#include <cuda.h>

__device__ double dist(const double, const double, const double, const double);
__global__ void timestep(const double *atoms_dev, const int atoms_sz, 
                         const double threshold, const double *vals_old,
                         double *vals_new, double *dists);

