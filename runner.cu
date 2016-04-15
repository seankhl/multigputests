
#include <vector>
#include <array>
#include <random>
#include <functional>
#include <iostream>
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

const int X_SZ = 100;
const int Y_SZ = 100;
const int RANGE = 32;
const int natoms = 100;
const int natompos = 200;

int run(int nprocs, int nt)
{
    //constexpr int X_NBINS = X_SZ / RANGE + 1;
    //constexpr int Y_NBINS = Y_SZ / RANGE + 1;
    //Grid grid = new Grid(X_NBINS, Y_NBINS);

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
    for (int i = 0; i < natompos; i+=2) {
        atom_pos[i+0] = x_rand();
        atom_pos[i+1] = y_rand();
        atom_val[i/2] = val_rand();
        //grid.record_atom(atoms[atoms.size() - 1]);
    }

    // device positions
    double *atom_pos_dev;
    HANDLE_ERROR( cudaMalloc((void **)&atom_pos_dev, natompos * sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy((void *)atom_pos_dev, (void *)atom_pos.data(), 
                             natompos * sizeof(double), cudaMemcpyHostToDevice) );

    // old vals
    double *atom_val_old_dev;
    HANDLE_ERROR( cudaMalloc((void **)&atom_val_old_dev, natoms * sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy((void *)atom_val_old_dev, (void *)atom_val.data(), 
                             natoms * sizeof(double), cudaMemcpyHostToDevice) );

    // new vals
    double *atom_val_new_dev;
    HANDLE_ERROR( cudaMalloc((void **)&atom_val_new_dev, natoms * sizeof(double)) );

    // record dists for funsies and debugging
    std::array<double, natoms> dists;
    double *dists_dev;
    HANDLE_ERROR( cudaMalloc((void **)&dists_dev, natoms * sizeof(double)) );

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
        //if (t % 10000 == 0) { std::cout << t << std::endl; }
        timestep<<<natoms, 1>>>(atom_pos_dev, natompos, 
                                RANGE, atom_val_old_dev,
                                atom_val_new_dev, dists_dev);
        timestep<<<natoms, 1>>>(atom_pos_dev, natompos, 
                                RANGE, atom_val_new_dev,
                                atom_val_old_dev, dists_dev);
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
    HANDLE_ERROR( cudaMemcpy((void *)atom_pos.data(), (void *)atom_pos_dev, 
                  natompos * sizeof(double), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy((void *)atom_val.data(), (void *)atom_val_new_dev, 
                  natoms * sizeof(double), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy((void *)dists.data(), (void *)dists_dev, 
                  natoms * sizeof(double), cudaMemcpyDeviceToHost) );

    std::cout << "atom 48: " << atom_pos[48*2+0] << " " 
                             << atom_pos[48*2+1] << " "
                             << atom_val[48] << " "
                             << dists[48] << std::endl;
	for (int i = 0; i < natoms; ++i) {
		std::cout << dists[i] << std::endl;
	}

    return 0;
}

int main() {
    return run(1, 1);
}

