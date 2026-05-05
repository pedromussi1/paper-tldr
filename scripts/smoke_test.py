"""Quick environment + GPU smoke test. Run after install to confirm:
- torch sees CUDA and the right GPU
- bitsandbytes loads (Windows wheel, sm_89 Ada Lovelace support)
- the rest of the ML stack imports cleanly
- a small CUDA tensor can actually compute
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible to torch")
        return 1

    print(f"CUDA built: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.2f} GB")
    print(f"compute capability: sm_{props.major}{props.minor}")

    import bitsandbytes as bnb
    print(f"bitsandbytes: {bnb.__version__}")

    import accelerate
    import datasets
    import evaluate
    import peft
    import transformers
    print(f"transformers: {transformers.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"accelerate: {accelerate.__version__}")
    print(f"datasets: {datasets.__version__}")
    print(f"evaluate: {evaluate.__version__}")

    # Allocate something nontrivial on GPU and force a kernel launch.
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() / 1e6
    print(f"matmul 2048x2048 fp16 OK, |c.sum|={c.sum().abs().item():.2f}, allocated: {used:.1f} MB")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
