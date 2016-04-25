
#include "sim.cuh"

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

/*
// square<T> computes the square of a number f(x) -> x*x 
template <typename T> 
struct square 
{ 
    __host__ __device__ T operator()(const T &x) const { return x * x; } 
};

float dist(Atom a, Atom b)
{
    thrust::device_vector<float> diff(2);
    thrust::transform(a.begin(), a.end(), b.begin(), b.end(), diff.begin(),
                      trust::minus<float>());
    return std::sqrt( thrust::transform_reduce(
                        diff.begin(), diff.end(), square<float>(),
                        0, thrust::plus<float>()) ); 
}
*/
__device__ float dist(const float ax, const float ay, 
                       const float bx, const float by)
{
    return sqrt(pow(ax - bx, 2) + pow(ay - by, 2));
}

__global__ void timestep(const int atoms_sz, const float threshold, 
                         const float4 *vals_old, float4 *vals_new,
                         const float4 *ghost_lo, const int ghost_lo_sz,
                         const float4 *ghost_hi, const int ghost_hi_sz)
{
    int i = blockIdx.x;
    int count = 0;
    vals_new[i].z = 0.0;
    for (int j = 0; j < atoms_sz; ++j) {
        vals_new[i].w = dist(vals_old[i].x, vals_old[i].y, 
                             vals_old[j].x, vals_old[j].y);
        if (vals_new[i].w < threshold) {
            vals_new[i].z += vals_old[j].z;
            ++count;
        }
    }
    for (int j = 0; j < ghost_lo_sz; ++j) {
        vals_new[i].w = dist(vals_old[i].x, vals_old[i].y, 
                             ghost_lo[j].x, ghost_lo[j].y);
        if (vals_new[i].w < threshold) {
            vals_new[i].z += ghost_lo[j].z;
            ++count;
        }
    }
    for (int j = 0; j < ghost_hi_sz; ++j) {
        vals_new[i].w = dist(vals_old[i].x, vals_old[i].y, 
                             ghost_hi[j].x, ghost_hi[j].y);
        if (vals_new[i].w < threshold) {
            vals_new[i].z += ghost_hi[j].z;
            ++count;
        }
    }
    vals_new[i].z /= count;
}

