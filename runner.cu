
#include <vector>
#include <array>
#include <random>
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

const int X_SZ = 1000;
const int Y_SZ = 1000;
const int RANGE = 32;
const int natoms = 10000;
const int natompos = 20000;

int run(int ngpus_in, int nt)
{
    //constexpr int X_NBINS = X_SZ / RANGE + 1;
    //constexpr int Y_NBINS = Y_SZ / RANGE + 1;
    //Grid grid = new Grid(X_NBINS, Y_NBINS);

    int ngpus;
    cudaGetDeviceCount(&ngpus);

    if (ngpus_in < ngpus) { ngpus = ngpus_in; }

    /* Check capability of the GPU 
        (should be done for each card to be used)
    */
    cudaDeviceProp prop1, prop2;

    // second argument is gpu number
    cudaGetDeviceProperties(&prop1, 1);
    cudaGetDeviceProperties(&prop2, 2);

    // check results
    bool is_fermi = false;
    if (prop1.major >= 2 && prop2.major >= 2)  is_fermi = true; // must be Fermi based
    std::cout << "num devices: " << ngpus << "; is fermi? " << is_fermi << std::endl;
    //return 0;

    // random initial values for atoms
    std::random_device rd;
    std::mt19937 mt_rand(rd());
    std::mt19937::result_type x_seed = time(0);
    auto x_rand = std::bind(
                        std::uniform_real_distribution<double>(0, X_SZ), 
                        std::mt19937(rd()));
    std::mt19937::result_type y_seed = time(0);
    auto y_rand = std::bind(
                        std::uniform_real_distribution<double>(0, Y_SZ), 
                        std::mt19937(rd()));
    std::mt19937::result_type val_seed = time(0);
    auto val_rand = std::bind(
                        std::uniform_real_distribution<double>(0, 1), 
                        std::mt19937(rd()));

    // create atoms and store them in the grid
    std::array<double, natompos> atom_pos;
    std::array<double, natompos> atom_val;
    for (int i = 0; i < natompos; i += 2) {
        atom_pos[i+0] = x_rand();
        atom_pos[i+1] = y_rand();
        atom_val[i/2] = val_rand();
        //grid.record_atom(atoms[atoms.size() - 1]);
    }
    std::array<double, natoms> dists;

    std::vector<double *> atom_pos_dev(ngpus);
    std::vector<double *> atom_val_old_dev(ngpus);
    std::vector<double *> atom_val_new_dev(ngpus);
    std::vector<double *> dists_dev(ngpus);
    int atompos_width = natompos / ngpus;
    int atoms_width = natoms / ngpus;
    int atompos_off = 0;
    int atoms_off = 0;
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i + 1);
        // device positions
        HANDLE_ERROR( cudaMalloc((void **)&atom_pos_dev[i], 
                                 atompos_width * sizeof(double)) );
        HANDLE_ERROR( cudaMemcpy((void *)atom_pos_dev[i], (void *)(atom_pos.data() + atompos_off), 
                                 atompos_width * sizeof(double), cudaMemcpyHostToDevice) );
        
        // old vals
        HANDLE_ERROR( cudaMalloc((void **)&atom_val_old_dev[i], 
                                 atoms_width * sizeof(double)) );
        HANDLE_ERROR( cudaMemcpy((void *)atom_val_old_dev[i], (void *)(atom_val.data() + atoms_off), 
                                 atoms_width * sizeof(double), cudaMemcpyHostToDevice) );
        
        // new vals
        HANDLE_ERROR( cudaMalloc((void **)&atom_val_new_dev[i], atoms_width * sizeof(double)) );
        
        // record dists for funsies and debugging
        HANDLE_ERROR( cudaMalloc((void **)&dists_dev[i], atoms_width * sizeof(double)) );

        atompos_off += atompos_width;
        atoms_off += atoms_width;
    }
    
    // timestep
    //int x_cell = 0;
    //int y_cell = 0;
    std::cout << "num timesteps: " << nt << std::endl;
    std::cout << "atom 48: " << atom_pos[48*2+0] << " " 
                             << atom_pos[48*2+1] << " "
                             << atom_val[48] << std::endl;
    for (int i = 0; i < natoms; ++i) {
        std::cout << atom_val[i] << std::endl;
    }

    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < ngpus; ++i) {
            cudaSetDevice(i + 1);
            //if (t % 10000 == 0) { std::cout << t << std::endl; }
            timestep<<<atoms_width, 1>>>(atom_pos_dev[i], atompos_width, 
                                    RANGE, atom_val_old_dev[i],
                                    atom_val_new_dev[i], dists_dev[i]);
            timestep<<<atoms_width, 1>>>(atom_pos_dev[i], atompos_width, 
                                    RANGE, atom_val_new_dev[i],
                                    atom_val_old_dev[i], dists_dev[i]);
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

    atompos_off = 0;
    atoms_off = 0;
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i + 1);
        HANDLE_ERROR( cudaMemcpy((void *)(atom_pos.data() + atompos_off), (void *)atom_pos_dev[i], 
                      atompos_width * sizeof(double), cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy((void *)(atom_val.data() + atoms_off), (void *)atom_val_new_dev[i], 
                      atoms_width * sizeof(double), cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy((void *)(dists.data() + atoms_off), (void *)dists_dev[i], 
                      atoms_width * sizeof(double), cudaMemcpyDeviceToHost) );
        atompos_off += atompos_width;
        atoms_off += atoms_width;
    }

    std::cout << "atom 48: " << atom_pos[48*2+0] << " " 
                             << atom_pos[48*2+1] << " "
                             << atom_val[48] << " "
                             << dists[48] << std::endl;
    for (int i = 0; i < natoms; ++i) {
        std::cout << dists[i] << std::endl;
    }
    
    for (int i = 0; i < ngpus; ++i) {
        cudaFree((void *)atom_pos_dev[i]);
        cudaFree((void *)atom_val_old_dev[i]);
        cudaFree((void *)atom_val_new_dev[i]);
        cudaFree((void *)dists_dev[i]);
    }

    return 0;
    
}

int main() {
    std::chrono::time_point<std::chrono::system_clock> one_start, one_end;
    one_start = std::chrono::system_clock::now();
    run(1, 1);
    one_end = std::chrono::system_clock::now();
    std::chrono::duration<double> one_dur = one_end - one_start;
    
    std::chrono::time_point<std::chrono::system_clock> two_start, two_end;
    two_start = std::chrono::system_clock::now();
    run(2, 1);
    two_end = std::chrono::system_clock::now();
    std::chrono::duration<double> two_dur = two_end - two_start;
    
    std::cout << "one took: " << one_dur.count() << " seconds; " << std::endl;
    std::cout << "two took: " << two_dur.count() << " seconds; " << std::endl;
}

