#include <mpi.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

// Run with: mpirun -np 4 ./hellow

// CUDA kernel function
__global__ void printFromDevice(int deviceId, double someValue) {
    printf("Hello World from GPU! I'm device %d with value %g\n", deviceId, someValue);
}

int main(int argc, char** argv) {

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if there are enough devices
    int deviceCount, deviceId;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount % size== 0) {
      // Set Device to the rank of the process
      deviceId = rank;
    }
    else {
      // Set Device all devices to use the same device
      deviceId = 0;
    }
    cudaSetDevice(deviceId);
    
    // Print message from every process
    printf("CPU process %d of %d is using device %d\n", rank, size, deviceId);

    // Generate random number at runtime
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double someValue = dis(gen);

    // Launch CUDA kernel and pass arguments by value
    printFromDevice<<<1, 1>>>(deviceId, someValue);
    cudaDeviceSynchronize();

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
