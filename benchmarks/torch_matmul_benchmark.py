import sys
import argparse
import time
import statistics
import json
import pathlib
# from pathlib import Path

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--dtype", type=str, choices=["float32"], default="float32")
    parser.add_argument("--output", type=str, default="result.json")
    args = parser.parse_args()
    if args.repeats <= 1 or args.warmup < 0 or args.n <= 0 or args.k <= 0 or args.m <= 0:
        print("arg invalid")
        sys.exit(1)
    return args

def run_matmul_cpu(x, y):
    xg = x.to("cuda")
    yg = y.to("cuda")
    z = torch.matmul(xg, yg).cpu()


def benchmark_gpu_execution(x, y, warmup, repeats):
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

def benchmark_transfer_inclusive(x, y, warmup, repeats):

    for _ in range(warmup):
        run_matmul_cpu(x, y)

    samples = []

    for _ in range(repeats):
        starts = time.perf_counter()
        run_matmul_cpu(x, y)
        elapsed_ms = (time.perf_counter() - starts) * 1000.0
        samples.append(elapsed_ms)
    return samples

def main():

    args = parse_args()

    dtype_map = {
        "float32":torch.float32
    }
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, dtype=dtype_map[args.dtype], device="cpu")
    y = torch.randn(args.k, args.n, dtype=dtype_map[args.dtype], device="cpu")

    # correncess
    z = torch.matmul(x, y)
    xg = x.to("cuda")
    yg = y.to("cuda")
    zg = torch.matmul(xg, yg).cpu()
    torch.testing.assert_close(zg, z, rtol=1e-5, atol=1e-4)

    # time
    samples1 = benchmark_transfer_inclusive(x, y, args.warmup, args.repeats)
    min1, median1 = min(samples1), statistics.median(samples1)

    samples2 = benchmark_gpu_execution(x, y, args.warmup, args.repeats)
    min2, median2 = min(samples2), statistics.median(samples2)

    result = {
        "configuration": {
            "m": args.m,
            "k": args.k,
            "n": args.n,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "correctness": {
            "passed": True,
            "rtol": 1e-5,
            "atol": 1e-4,
        },
        "timing_ms": {
            "gpu_execution": {
                "raw_samples": samples2,
                "min": min2,
                "median": median2,
            },
            "transfer_inclusive_end_to_end": {
                "raw_samples": samples1,
                "min": min1,
                "median": median1,
            },
        },
    }
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Result JSON: {output_path}")

if __name__ == "__main__":
    raise SystemExit(main())
