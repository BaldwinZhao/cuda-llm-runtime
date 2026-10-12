#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Minimal error checking is included so this complete example fails loudly.
// Chapter 4 will develop CUDA error handling more systematically.
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    const cudaError_t error = (call);                                                              \
    if (error != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,      \
                   cudaGetErrorString(error));                                                     \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (false)

// 同时生成 CPU 版本和 GPU 版本。
__host__ __device__ double add_scalar(const double x, const double y) { return x + y; }

// 由 CPU 发起调用，但由 GPU 上的大量线程并行执行。
__global__ void vector_add_kernel(const double *x, const double *y, double *z, const int count) {
  // 将 block 编号和 block 内线程编号组合成全局线程编号。
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  // grid_size 是向上取整得到的，最后一个 block 可能包含多余线程。
  if (index < count) {
    z[index] = add_scalar(x[index], y[index]);
  }
}

// CPU 参考实现，用于生成正确答案。
void vector_add_cpu(const std::vector<double> &x, const std::vector<double> &y,
                    std::vector<double> &expected) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    expected[i] = add_scalar(x[i], y[i]);
  }
}

// 比较 GPU 结果和 CPU 结果。
// 浮点数不适合直接使用 ==，因此同时使用绝对误差和相对误差。
bool check_result(const std::vector<double> &actual, const std::vector<double> &expected) {
  constexpr double absolute_tolerance = 1.0e-12;
  constexpr double relative_tolerance = 1.0e-12;

  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double difference = std::fabs(actual[i] - expected[i]);

    const double tolerance = absolute_tolerance + relative_tolerance * std::fabs(expected[i]);

    if (difference > tolerance) {
      std::fprintf(stderr, "Mismatch at index %zu: GPU=%.17g CPU=%.17g diff=%.3e\n", i, actual[i],
                   expected[i], difference);

      return false;
    }
  }

  return true;
}

int main() {
  // 故意选择一个不能被 256 整除的元素数量。
  // 这样可以验证 kernel 中的边界判断确实有效。
  constexpr int element_count = 1'000'003;

  // 一个 block 包含 256 个线程。
  constexpr int block_size = 256;

  // 向上取整：
  //
  // grid_size = ceil(element_count / block_size)
  //
  // 不能直接使用 element_count / block_size，
  // 否则最后不足一个 block 的元素不会被处理。
  const int grid_size = (element_count + block_size - 1) / block_size;

  // CUDA 内存接口接收的是字节数。
  const std::size_t bytes = static_cast<std::size_t>(element_count) * sizeof(double);

  // 选择编号为 0 的 GPU。
  CUDA_CHECK(cudaSetDevice(0));

  // 查询 GPU 基本信息。
  cudaDeviceProp properties{};

  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::printf("GPU: %s, compute capability %d.%d\n", properties.name, properties.major,
              properties.minor);

  // ============================================================
  // 1. 分配 Host 内存
  // ============================================================
  //
  // h_ 前缀表示这些数据位于主机内存中。

  std::vector<double> h_x(element_count);
  std::vector<double> h_y(element_count);
  std::vector<double> h_z(element_count, 0.0);
  std::vector<double> h_expected(element_count, 0.0);

  // 初始化输入数据。
  for (int i = 0; i < element_count; ++i) {
    h_x[i] = 1.0 + 0.001 * static_cast<double>(i % 1000);
    h_y[i] = 2.0 - 0.0005 * static_cast<double>(i % 1000);
  }

  // 使用 CPU 计算参考答案。
  vector_add_cpu(h_x, h_y, h_expected);

  // ============================================================
  // 2. 分配 Device 内存
  // ============================================================
  //
  // d_ 前缀表示这些指针指向 GPU 显存。

  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_z = nullptr;

  // cudaMalloc 需要修改指针变量本身，
  // 所以传入的是指针变量的地址，即 void**。
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), bytes));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), bytes));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_z), bytes));

  // ============================================================
  // 3. Host -> Device
  // ============================================================

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  std::printf("Launch: grid=%d blocks, block=%d threads, total=%d threads\n", grid_size, block_size,
              grid_size * block_size);

  // ============================================================
  // 4. 启动 Kernel
  // ============================================================
  //
  // <<<grid_size, block_size>>>
  //
  // grid_size：一共启动多少个 block
  // block_size：每个 block 启动多少个线程

  vector_add_kernel<<<grid_size, block_size>>>(d_x, d_y, d_z, element_count);

  // Kernel 启动是异步的。
  //
  // cudaGetLastError：
  // 检查启动配置、参数等即时错误。
  CUDA_CHECK(cudaGetLastError());

  // 等待 GPU 执行完成，并检查执行期间出现的错误。
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================
  // 5. Device -> Host
  // ============================================================

  CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));

  // ============================================================
  // 6. 正确性验证
  // ============================================================

  const bool passed = check_result(h_z, h_expected);

  std::printf("Samples: z[0]=%.6f, z[123456]=%.6f, z[last]=%.6f\n", h_z[0], h_z[123456],
              h_z[element_count - 1]);

  std::printf("Result: %s\n", passed ? "PASS" : "FAIL");

  // ============================================================
  // 7. 释放 Device 内存
  // ============================================================
  //
  // 每一次成功的 cudaMalloc 都必须对应一次 cudaFree。

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_z));

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
