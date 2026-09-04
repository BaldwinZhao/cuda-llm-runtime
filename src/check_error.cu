

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

bool check_error(cudaError_t status, const std::string error_str) {

  if (status != cudaSuccess) {
    std::cerr << error_str << cudaGetErrorString(status) << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
