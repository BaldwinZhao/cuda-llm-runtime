import sys
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()
    if args.m <= 0 or args.k <=0 or args.n <= 0:
        parser.error("--m --n --k must be positive")
    return args

def make_inputs(m, k, n, seed):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.float32, device="cpu")
    y = torch.randn((k, n), dtype=torch.float32, device="cpu")
    return x, y

def run_reference(x, y):
    return torch.matmul(x, y)

def run_candidate(x, y):
    xg = x.to("cuda")
    yg = y.to("cuda")
    return torch.matmul(xg, yg).cpu()

def check_correctness(reference, candidate, rtol, atol):
    torch.testing.assert_close(reference, candidate, rtol=rtol, atol=atol)


def self_test():
    x, y = make_inputs(2, 2, 2, 2)
    z = torch.matmul(x, y)
    x[0][0] = x[0][0] + 1.0
    xg = x.to("cuda")
    yg = y.to("cuda")
    z_can = torch.matmul(xg, yg).cpu()

    try:
        check_correctness(z, z_can, rtol=1e-5, atol=1e-4)
    except AssertionError:
        print("self-test: PASS (known bad result was rejected)")
    else:
       raise RuntimeError("self-test: FAIL (validator accepted a bad result)")

def main():
    args = parse_args()
    print("python: ", sys.version)
    print("torch: ", torch.__version__)
    print("cuda available: ", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return 1
    print("GPU: ", torch.cuda.get_device_name(0))

    # TODO 1: make input
    x, y = make_inputs(args.m, args.k, args.n, args.seed)

    # TODO 2: run_reference
    z = run_reference(x, y)

    # TODO 3: run_candidate
    z_can = run_candidate(x, y)

    # TODO 4: compare
    try:
        check_correctness(z, z_can, 1e-5, 1e-4)
    except AssertionError:
        raise RuntimeError("test: failed")
    # TODO 5: self test
    if (args.self_test):
        self_test()
    # print("input1: ", x_gpu.shape, x_gpu.dtype, x_gpu.device)
    # print("input2: ", y_gpu.shape, y_gpu.dtype, y_gpu.device)
    # print("output1: ", out_gpu.shape, out_gpu.dtype, out_gpu.device)

if __name__ == "__main__":
    raise SystemExit(main())
