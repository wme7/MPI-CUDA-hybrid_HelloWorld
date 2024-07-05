#include <mpi.h>
#include <iostream>
#include <cuda_runtime.h>

// Run with: mpirun -np 2 ./hellow

// CUDA kernel function
__global__ void printFromDevice(int deviceId, double somevalue) {
    printf("Hello World from GPU! I'm device %d with value %g\n", deviceId, somevalue);
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

    // Launch CUDA kernel
    double somevalue = 3.1416;
    printFromDevice<<<1, 1>>>(deviceId, somevalue);
    cudaDeviceSynchronize();

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
