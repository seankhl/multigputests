
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

double dist(Atom a, Atom b)
{
    thrust::device_vector<double> diff(2);
    thrust::transform(a.begin(), a.end(), b.begin(), b.end(), diff.begin(),
                      trust::minus<double>());
    return std::sqrt( thrust::transform_reduce(
                        diff.begin(), diff.end(), square<double>(),
                        0, thrust::plus<double>()) ); 
}
*/
__device__ double dist(const double ax, const double ay, 
					   const double bx, const double by)
{
    return sqrt(pow(ax - bx, 2) + pow(ay - by, 2));
}

__global__ void timestep(const double *atoms_dev, const int atoms_sz, 
                         const double threshold, const double *vals_old,
                         double *vals_new, double *dists)
{
    int i = blockIdx.x * 2;
    int count = 0;
    for (int j = 0; j < atoms_sz; j += 2) {
        dists[blockIdx.x] = dist(atoms_dev[i+0], atoms_dev[i+1], 
                                 atoms_dev[j+0], atoms_dev[j+1]);
        if (dists[blockIdx.x] < threshold) {
            vals_new[blockIdx.x] += vals_old[j/2];
            ++count;
        }
    }
    vals_new[blockIdx.x] /= count;
}

