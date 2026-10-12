

#include <cstdio>
#include <iostream>

__global__ void hello() {
  printf("block index = %d, thread index = %d\n", blockIdx.x, threadIdx.x);
}

__global__ void mul_n(float n, const float *input, float *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  output[i] = input[i] * n;
}

int main() {

  hello<<<2, 128>>>();
  cudaDeviceSynchronize();
  // mul_n(2.0, const float *input, float *output)
  return 0;
}
