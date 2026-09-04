
#include "check_error.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define VALUE 43

__global__ void write_value(int *output) { output[0] = VALUE; }

__global__ void write_value(int *output, int value) { output[0] = value; }

int main() {

  int *device_output = nullptr;
  int host_output = 0;

  cudaError_t status = cudaMalloc(&device_output, sizeof(*device_output));

  if (check_error(status, "cudaMalloc Failed:")) {
    return EXIT_FAILURE;
  }

  // original
  // write_value<<<1, 1>>>(device_output);
  // work 1
  write_value<<<1, 1>>>(device_output, VALUE);
  // work 2 : 44 != VALUE
  // write_value<<<1, 1>>>(device_output, 44);

  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(&host_output, device_output, sizeof(int), cudaMemcpyDeviceToHost);
  }

  const cudaError_t free_status = cudaFree(device_output);

  if (check_error(status, "cuda failed. ")) {
    return EXIT_FAILURE;
  }

  if (check_error(free_status, "cuda failed. ")) {
    return EXIT_FAILURE;
  }

  if (host_output != VALUE) {
    std::cerr << "Unexpected result : " << host_output << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "cuda success: " << host_output << '\n';
  return EXIT_SUCCESS;
}
