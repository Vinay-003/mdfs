
import argparse
import csv
import math
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gpu",
    required=True
)

parser.add_argument(
    "--input",
    required=True
)

parser.add_argument(
    "--output",
    required=True
)

args = parser.parse_args()


# IMPORTANT:
# Must happen BEFORE importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


import pandas as pd
import torch

from src.mdfs_scorer import MDFSScorer


print("=" * 60)
print(f"PHYSICAL GPU ASSIGNED: {args.gpu}")
print("CUDA_VISIBLE_DEVICES:",
      os.environ["CUDA_VISIBLE_DEVICES"])
print("Visible CUDA devices:",
      torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

print(
    "Visible GPU:",
    torch.cuda.get_device_name(0)
)
print("=" * 60)


manifest = pd.read_csv(args.input)

output_path = Path(args.output)

completed = set()

if output_path.exists():

    try:
        existing = pd.read_csv(output_path)

        if "filename" in existing.columns:
            completed = set(
                existing["filename"].astype(str)
            )

    except Exception:
        completed = set()


print(
    f"Worker GPU {args.gpu}: "
    f"{len(manifest)} assigned"
)

print(
    f"Worker GPU {args.gpu}: "
    f"{len(completed)} already completed"
)


scorer = MDFSScorer(
    weights_path=
        "artifacts/MDFS_weights_rebuilt.pth",
    device="cuda"
)


fieldnames = list(manifest.columns) + [
    "mdfs_raw_score",
    "runtime_seconds",
    "worker_gpu"
]


write_header = (
    not output_path.exists()
    or output_path.stat().st_size == 0
)


start_all = time.perf_counter()

processed_now = 0


with output_path.open(
    "a",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    if write_header:
        writer.writeheader()


    for _, row in manifest.iterrows():

        filename = str(
            row["filename"]
        )

        if filename in completed:
            continue


        torch.cuda.synchronize()

        start = time.perf_counter()


        score = scorer.score_path(
            row["image_path"]
        )


        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )


        if not math.isfinite(score):
            raise RuntimeError(
                f"Non-finite score: "
                f"{filename} = {score}"
            )


        record = row.to_dict()

        record.update({
            "mdfs_raw_score":
                score,

            "runtime_seconds":
                elapsed,

            "worker_gpu":
                int(args.gpu),
        })


        writer.writerow(record)

        # Persist every completed image
        f.flush()

        processed_now += 1


        if (
            processed_now % 50 == 0
            or processed_now
            == len(manifest)
        ):

            print(
                f"[GPU {args.gpu}] "
                f"{processed_now}/"
                f"{len(manifest)} "
                f"{filename} "
                f"MDFS={score:.6f} "
                f"{elapsed:.4f}s",
                flush=True
            )


total_elapsed = (
    time.perf_counter()
    - start_all
)


result = pd.read_csv(
    output_path
)


print()
print("=" * 60)
print(
    f"GPU {args.gpu} WORKER COMPLETE"
)
print("=" * 60)

print(
    "Rows:",
    len(result)
)

print(
    "Unique filenames:",
    result["filename"].nunique()
)

print(
    "Mean inference time:",
    result["runtime_seconds"].mean()
)

print(
    "Worker wall time:",
    total_elapsed
)

print(
    "All finite:",
    result[
        "mdfs_raw_score"
    ].map(
        math.isfinite
    ).all()
)
