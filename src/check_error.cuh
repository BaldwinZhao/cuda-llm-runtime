
#pragma once

#include <cuda_runtime.h>
#include <string>

extern bool check_error(cudaError_t status, const std::string);
