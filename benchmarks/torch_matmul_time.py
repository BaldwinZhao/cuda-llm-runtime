import sys
import argparse
import time
import statistics

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    args = parser.parse_args()
    if args.repeats <= 1 or args.warmup < 0 or args.n <= 0 or args.k <= 0 or args.m <= 0:
        print("arg invalid")
        sys.exit(1)
    return args

def benchmark_gpu_execution(x, y, warmup, repeats):
    # Inputs already on CUDA Device.
    # Return raw milliseconds.
    xg = x.to("cuda")
    yg = y.to("cuda")
    for _ in range(warmup):
        torch.matmul(xg, yg)

    samples = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch.matmul(xg, yg)
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
    return samples

def run_matmul_cpu(x, y):
    xg = x.to("cuda")
    yg = y.to("cuda")
    z = torch.matmul(xg, yg).cpu()

def benchmark_transfer_inclusive(x, y, warmup, repeats):
    # Inputs start on CPU; each sample includes H2D, Matmul and D2H.
    # Return raw milliseconds.
    # warmup
    for _ in range(warmup):
        run_matmul_cpu(x, y)

    samples = []
    # test
    for _ in range(repeats):
        start_s = time.perf_counter()
        run_matmul_cpu(x, y)
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        samples.append(elapsed_ms)
    return samples

def summarize(samples):
    return min(samples), statistics.median(samples)


def main():
    args = parse_args()
    x = torch.randn(args.m, args.k, dtype=torch.float32, device="cpu")
    y = torch.randn(args.k, args.n, dtype=torch.float32, device="cpu")

    samples1 = benchmark_transfer_inclusive(x, y, args.warmup, args.repeats)
    min1, median1 = summarize(samples1)
    print(samples1)
    print("transfer_inclusive: ", min1, ", ", median1)

    samples2 = benchmark_gpu_execution(x, y, args.warmup, args.repeats)
    min2, median2 = summarize(samples2)
    print(samples2)
    print("gpu execution: ", min2, ", ", median2)

if __name__ == "__main__":
    raise SystemExit(main())
