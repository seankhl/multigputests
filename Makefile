
NVCC_DIR = /usr/local/cuda
OMPI_DIR = /usr/local/mpi-cuda
CXX_BIN = g++

NVCC_INC = -I$(NVCC_DIR)/include
OMPI_INC = -I$(OMPI_DIR)/include

OMPI_LINK = -L$(OMPI_DIR)/lib

CXX_FLAGS = -std=c++11 -g

gpudirect:
	$(NVCC_DIR)/bin/nvcc $(CXX_FLAGS) -c runner.cu sim.cu
	$(NVCC_DIR)/bin/nvcc $(CXX_FLAGS) runner.o sim.o -o gpud_test

mpi:
	$(NVCC_DIR)/bin/nvcc $(CXX_FLAGS) $(OMPI_INC) $(OMPI_LINK) -lmpi -c runner_mpi.cu sim.cu
	$(NVCC_DIR)/bin/nvcc $(CXX_FLAGS) $(OMPI_INC) $(OMPI_LINK) -lmpi runner_mpi.o sim.o -o mpi_test

.PHONY: clean
clean:
	@$(RM) *.o gpud_test mpi_test

