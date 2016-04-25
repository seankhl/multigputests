
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <iostream>
#include <chrono>
//#include "Eigen/dense"

#include "sim.cuh"

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

const int X_SZ = 2048;
const int Y_SZ = 2048;
const float RANGE = 32;
const int natoms = 1000;

void checkDevices(int ngpus) {
    /* Check capability of the GPU 
       (should be done for each card to be used)
    */
    std::vector<cudaDeviceProp> gpuprops(ngpus);
    bool is_fermi = true;
    bool has_uva = true;
    // second argument is gpu number
    for (int i = 0; i < ngpus; ++i) {
        cudaGetDeviceProperties(&gpuprops[i], i + 1);
    }
    // check results
    for (int i = 0; i < ngpus; ++i) {
        is_fermi &= (gpuprops[i].major >= 2); // must be Fermi based
        has_uva &= (gpuprops[i].unifiedAddressing); // must be Fermi based
    }
    
    // TODO: only works for ngpus == 2
    int access2from1, access1from2;
    cudaDeviceCanAccessPeer(&access2from1, 2, 1);
    cudaDeviceCanAccessPeer(&access1from2, 1, 2);
    bool same_complex = (access2from1 && access1from2);
    std::cout << "num devices: " << ngpus << 
                 "; is fermi? " << is_fermi << 
                 "; has uva? " << has_uva <<
                 "; same complex? " << same_complex << std::endl;
}

int run(int ngpus_in, int nt)
{
    //constexpr int X_NBINS = X_SZ / RANGE + 1;
    //constexpr int Y_NBINS = Y_SZ / RANGE + 1;
    //Grid grid = new Grid(X_NBINS, Y_NBINS);

    int ngpus;
    cudaGetDeviceCount(&ngpus);

    if (ngpus_in < ngpus) { ngpus = ngpus_in; }
    checkDevices(ngpus);

    // random initial values for atoms
    std::random_device rd;
    std::mt19937 mt_rand(rd());
    std::mt19937::result_type x_seed = time(0);
    auto x_rand = std::bind(
                        std::uniform_real_distribution<float>(0, X_SZ), 
                        std::mt19937(1));
    std::mt19937::result_type y_seed = time(0);
    auto y_rand = std::bind(
                        std::uniform_real_distribution<float>(0, Y_SZ), 
                        std::mt19937(2));
    std::mt19937::result_type val_seed = time(0);
    auto val_rand = std::bind(
                        std::uniform_real_distribution<float>(0, 1), 
                        std::mt19937(3));

    // create atoms and store them in the grid
    std::array<float4, natoms> atoms;
    for (int i = 0; i < natoms; ++i) {
        atoms[i].x = x_rand();
        atoms[i].y = y_rand();
        atoms[i].z = val_rand();
        //grid.record_atom(atoms[atoms.size() - 1]);
    }

    std::sort(atoms.begin(), atoms.end(), [](float4 a, float4 b) {
        return a.y < b.y;
    });

    int atoms_off = 0;
    
    std::vector<int> cutlo(ngpus, 0);
    std::vector<int> atoms_split(ngpus, natoms);
    std::vector<int> atoms_width(ngpus, natoms);
    std::vector<int> cuthi(ngpus, natoms);
    for (int i = 0; i < ngpus; ++i) {
        for (int j = 0; j < natoms; ++j) {
            if (atoms[j].y > Y_SZ/ngpus * (i+1)) {
                atoms_split[i] = j;
                atoms_width[i] = j;
                break;
            }
        }
    }
    for (int i = 1; i < ngpus; ++i) {
        atoms_width[i] -= atoms_width[i-1];
    }
    for (int i = 0; i < ngpus; ++i) {
        for (int j = 0; j < natoms; ++j) {
            if (atoms[j].y > (Y_SZ/ngpus - RANGE) * (i+1)) {
                cutlo[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < ngpus; ++i) {
        for (int j = 0; j < natoms; ++j) {
            if (atoms[j].y > (Y_SZ/ngpus + RANGE) * (i+1)) {
                cuthi[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < ngpus; ++i) {
        std::cout << "gpu id: " << i+1 << 
                     " cutlo: " << cutlo[i] <<
                     " split: " << atoms_split[i] <<
                     " cuthi: " << cuthi[i] << std::endl;
    }

    std::vector<float4 *> atoms_old_dev(ngpus);
    std::vector<float4 *> atoms_new_dev(ngpus);
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i + 1);
        
        // new vals: before timestepping, current vals always in here
        HANDLE_ERROR( cudaMalloc((void **)&atoms_new_dev[i], 
                                 atoms_width[i] * sizeof(float4)) );
        HANDLE_ERROR( cudaMemcpy((void *)atoms_new_dev[i], 
                                 (void *)(atoms.data() + atoms_off), 
                                 atoms_width[i] * sizeof(float4), 
                                 cudaMemcpyHostToDevice) );
        
        // malloc space for old vals
        HANDLE_ERROR( cudaMalloc((void **)&atoms_old_dev[i], 
                                 atoms_width[i] * sizeof(float4)) );
        HANDLE_ERROR( cudaMemcpy((void *)atoms_old_dev[i], 
                                 (void *)(atoms.data() + atoms_off), 
                                 atoms_width[i] * sizeof(float4), 
                                 cudaMemcpyHostToDevice) );
        
        
        atoms_off += atoms_width[i];
    }
    
    std::vector<float4 *> ghost_lo_dev(ngpus-1);
    std::vector<float4 *> ghost_hi_dev(ngpus-1);
    for (int i = 0; i < ngpus-1; ++i) {  // don't need last split, always end
        // ghost vals
        cudaSetDevice(i + 2);  // next proc gets lo ghosts
        HANDLE_ERROR( cudaMalloc((void **)&ghost_lo_dev[i], 
                                 (atoms_split[i] - cutlo[i]) * sizeof(float4)) );
        HANDLE_ERROR( cudaMemcpy((void *)ghost_lo_dev[i], 
                                 (void *)(atoms.data() + cutlo[i]), 
                                 (atoms_split[i] - cutlo[i]) * sizeof(float4), 
                                 cudaMemcpyHostToDevice) );
        cudaSetDevice(i + 1);  // this proc gets hi ghosts
        HANDLE_ERROR( cudaMalloc((void **)&ghost_hi_dev[i], 
                                 (cuthi[i] - atoms_split[i]) * sizeof(float4)) );
        HANDLE_ERROR( cudaMemcpy((void *)ghost_hi_dev[i], 
                                 (void *)(atoms.data() + atoms_split[i]), 
                                 (cuthi[i] - atoms_split[i]) * sizeof(float4), 
                                 cudaMemcpyHostToDevice) );
    }
    
    // timestep
    //int x_cell = 0;
    //int y_cell = 0;
    std::cout << "num timesteps: " << nt << std::endl;
    for (int i = 0; i < natoms; i += natoms/10) {
        std::cout << atoms[i].x << " " << 
                     atoms[i].y << " " << 
                     atoms[i].z << std::endl;
    }

    float4 *needs_lo;
    int needs_lo_sz;
    float4 *needs_hi;
    int needs_hi_sz;
    float4 *atoms_tmp_dev;
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < ngpus; ++i) {
            cudaSetDevice(i + 1);
            //if (t % 10000 == 0) { std::cout << t << std::endl; }

            // figure out what our needed ghosts are
            if (i == 0) {
                needs_lo = NULL;
                needs_lo_sz = 0;
            } else {
                needs_lo = ghost_lo_dev[i-1];
                needs_lo_sz = atoms_split[i-1] - cutlo[i-1];
            }
            if (i == ngpus-1) {
                needs_hi = NULL;
                needs_hi_sz = 0;
            } else {
                needs_hi = ghost_hi_dev[i];
                needs_hi_sz = cuthi[i] - atoms_split[i];
            }
            
            // swap old and new pointers
            atoms_tmp_dev = atoms_new_dev[i];
            atoms_new_dev[i] = atoms_old_dev[i];
            atoms_old_dev[i] = atoms_tmp_dev;

            // run sim
            timestep<<<atoms_width[i], 1>>>(
                        atoms_width[i], RANGE, 
                        atoms_old_dev[i], atoms_new_dev[i],
                        needs_lo, needs_lo_sz,
                        needs_hi, needs_hi_sz);

            cudaDeviceSynchronize();

            // update ghosts
            if (i != 0) {
                int ghost_lo_sz = atoms_split[i-1] - cutlo[i-1];
                if (ghost_lo_sz != 0) {  // i != ngpus-1
                    HANDLE_ERROR( 
                        cudaMemcpy(
                            (void *)ghost_lo_dev[i-1], 
                            (void *)(atoms_new_dev[i-1] + 
                                        (atoms_width[i-1] - ghost_lo_sz)), 
                            ghost_lo_sz * sizeof(float4), 
                            cudaMemcpyDeviceToDevice) );
                }
                int ghost_hi_sz = cuthi[i-1] - atoms_split[i-1];
                if (cuthi[i-1] - atoms_split[i-1] != 0) {  // i != 0
                    HANDLE_ERROR( 
                        cudaMemcpy(
                            (void *)ghost_hi_dev[i-1], 
                            (void *)(atoms_new_dev[i]), 
                            ghost_hi_sz * sizeof(float4), 
                            cudaMemcpyDeviceToDevice) );
                }
            }

        }
        /*
        for (auto atom: atoms) {
            grid.get_cell_by_atom(atom, x_cell, y_cell);
            for (int x_off = -1; x_off <= 1; ++x_off) {
                for (int y_off = -1; y_off <= 1; ++y_off) {
                    vector<int> neighbors = grid.get_atoms_in_cell(
                                                x_cell + x_off, 
                                                y_cell + y_off);
                    if (x_off != 0 && y_off != 0) {
                        neighbors
        */
    }
    /*
            timestep<<<atoms_width[i], 1>>>(
                        atoms_width[i], RANGE, 
                        atoms_new_dev[i], atoms_old_dev[i],
                        ghost_lo_dev[i], atoms_split[i] - cutlo[i],
                        ghost_hi_dev[i], cuthi[i] - atoms_split[i]);
    */

    atoms_off = 0;
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i + 1);
        HANDLE_ERROR( cudaMemcpy((void *)(atoms.data() + atoms_off), 
                                 (void *)atoms_new_dev[i], 
                                 atoms_width[i] * sizeof(float4), 
                                 cudaMemcpyDeviceToHost) );
        atoms_off += atoms_width[i];
    }
    
    std::cout << "results: " << std::endl;
    for (int i = 0; i < natoms; i += natoms/10) {
        std::cout << atoms[i].x << " " << 
                     atoms[i].y << " " << 
                     atoms[i].z << " " <<
                     atoms[i].w << std::endl;
    }

    for (int i = 0; i < ngpus; ++i) {
        cudaFree((void *)atoms_old_dev[i]);
        cudaFree((void *)atoms_new_dev[i]);
    }

    return 0;
    
}

int main() {
    std::chrono::time_point<std::chrono::steady_clock> one_start, one_end;
    one_start = std::chrono::steady_clock::now();
    run(2, 10);
    one_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> one_dur = one_end - one_start;
    
    std::chrono::time_point<std::chrono::steady_clock> two_start, two_end;
    two_start = std::chrono::steady_clock::now();
    run(1, 10);
    two_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> two_dur = two_end - two_start;
    
    std::cout << "one took: " << one_dur.count() << " seconds; " << std::endl;
    std::cout << "two took: " << two_dur.count() << " seconds; " << std::endl;
}

