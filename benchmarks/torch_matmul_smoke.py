import sys
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    return args

def main():
    args = parse_args()
    print("python: ", sys.version)
    print("torch: ", torch.__version__)
    print("cuda available: ", torch.cuda.is_available())

    if not torch.cuda.is_available():
        return 1
    print("GPU: ", torch.cuda.get_device_name(0))

    torch.manual_seed(args.seed)

    # TODO 1: 在 CPU 创建两个 shape=[size, size]、dtype=float32 的 Tensor。
    x = torch.randn((args.size, args.size), dtype=torch.float32, device="cpu")
    y = torch.randn((args.size, args.size), dtype=torch.float32, device="cpu")
    # TODO 2: 把两个 Tensor 移动到 CUDA Device。
    x_gpu = x.to("cuda")
    y_gpu = y.to("cuda")
    # TODO 3: 使用 torch.matmul 或 @ 完成 Matmul。
    out_gpu = torch.matmul(x_gpu, y_gpu)
    # TODO 4: 打印输入和输出的 shape、dtype、device。
    print("input1: ", x_gpu.shape, x_gpu.dtype, x_gpu.device)
    print("input2: ", y_gpu.shape, y_gpu.dtype, y_gpu.device)
    print("output1: ", out_gpu.shape, out_gpu.dtype, out_gpu.device)

if __name__ == "__main__":
    raise SystemExit(main())
