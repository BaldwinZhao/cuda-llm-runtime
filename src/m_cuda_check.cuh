
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace AngryZ {

inline void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {

  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ":" +
                           std::string(expr) + ":" + std::string(cudaGetErrorString(status)));
}

} // namespace AngryZ

#define CUDA_CHECK(expr) AngryZ::check_cuda(expr, #expr, __FILE__, __LINE__)
