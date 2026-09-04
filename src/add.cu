

#include "m_cuda_check.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256
__global__ void device_add(float *a, float *b, size_t size, float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) {
    return;
  }
  c[i] = a[i] + b[i];
  // for (size_t i = 0; i < size; ++i) {
  //   c[i] = a[i] + b[i];
  // }
}

void scalar_add(float *a, float *b, size_t size, float *c) {
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "argc != 2\n";
    return EXIT_FAILURE;
  }
  std::string argv1(argv[1]);
  std::size_t consumed = 0;
  int temp_n{};
  try {
    temp_n = std::stoi(argv[1], &consumed);
  } catch (const std::exception &exception) {
    std::cerr << "failed to parse N: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  if (consumed != argv1.size() || temp_n <= 0) {
    std::cerr << "input arg1 is invalid\n";
    return EXIT_FAILURE;
  }
  std::size_t n = temp_n;
  std::cout << n << std::endl;

  // host 完成测试数据的生成
  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> c(n);
  for (int i = 0; i < n; ++i) {
    a[i] = 1.0f * i;
    b[i] = 1.0f * i;
  }

  // host 完成 vector add，记录运行结果
  scalar_add(a.data(), b.data(), n, c.data());

  for (int i = 0; i < n; ++i) {
    std::cout << c[i] << ", ";
  }
  std::cout << '\n';
  // device 内存分配
  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  std::vector<float> device_output_cmp(n);

  try {
    CUDA_CHECK(cudaMalloc(&device_a, n * sizeof(*device_a)));
    CUDA_CHECK(cudaMalloc(&device_b, n * sizeof(*device_a)));
    CUDA_CHECK(cudaMalloc(&device_c, n * sizeof(*device_a)));

    // host 测试数据写入到 Device
    CUDA_CHECK(cudaMemcpy(device_a, a.data(), n * sizeof(float), ::cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, b.data(), n * sizeof(float), ::cudaMemcpyHostToDevice));

    // device vector add kernel run
    // cudaDeviceProp prop{};
    // CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::size_t block_size = BLOCK_SIZE;
    std::size_t grid_count = (n + block_size - 1) / block_size;
    device_add<<<grid_count, block_size>>>(device_a, device_b, n, device_c);

    // check Launch Error
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
      status = cudaDeviceSynchronize();
    }

    if (status != cudaSuccess) {
      throw std::runtime_error("cudaDeviceSynchronize failed");
    }

    // device to host
    CUDA_CHECK(
        cudaMemcpy(device_output_cmp.data(), device_c, n * sizeof(a[0]), ::cudaMemcpyDeviceToHost));
  } catch (const std::exception &exception) {
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return EXIT_FAILURE;
  }
  // 对比结果
  // tolerence：目前暂时设置0，完全相等
  for (size_t i = 0; i < n; ++i) {
    if (c[i] != device_output_cmp[i]) {
      std::cerr << "not equal: " << c[i] << ", " << device_output_cmp[i] << std::endl;
      cudaFree(device_a);
      cudaFree(device_b);
      cudaFree(device_c);
      return EXIT_FAILURE;
    }
  }
  std::cout << "equal\n";

  // 释放内存
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return EXIT_SUCCESS;
}
