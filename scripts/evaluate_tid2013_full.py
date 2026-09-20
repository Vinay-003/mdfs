
import csv
import math
import time
from pathlib import Path

import pandas as pd
import torch

from src.mdfs_scorer import MDFSScorer

MANIFEST_PATH = "results/tid2013_manifest.csv"
OUTPUT_PATH = "results/tid2013_full_scores.csv"
WEIGHTS_PATH = "artifacts/MDFS_weights_rebuilt.pth"

manifest = pd.read_csv(MANIFEST_PATH)

scorer = MDFSScorer(
    weights_path=WEIGHTS_PATH,
    device="cuda"
)

output_path = Path(OUTPUT_PATH)

completed = set()

if output_path.exists():
    existing = pd.read_csv(output_path)
    if "filename" in existing.columns:
        completed = set(existing["filename"].astype(str))

print("Already completed:", len(completed))
print("Total to evaluate:", len(manifest))

fieldnames = [
    "filename",
    "image_path",
    "mos",
    "reference_id",
    "distortion_id",
    "distortion_level",
    "mdfs_raw_score",
    "runtime_seconds"
]

write_header = not output_path.exists() or output_path.stat().st_size == 0

start_all = time.perf_counter()

with output_path.open("a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    if write_header:
        writer.writeheader()

    for idx, row in manifest.iterrows():

        filename = str(row["filename"])

        if filename in completed:
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        score = scorer.score_path(
            row["image_path"]
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if not math.isfinite(score):
            raise RuntimeError(
                f"Non-finite score for {filename}: {score}"
            )

        writer.writerow({
            "filename": filename,
            "image_path": row["image_path"],
            "mos": row["mos"],
            "reference_id": row["reference_id"],
            "distortion_id": row["distortion_id"],
            "distortion_level": row["distortion_level"],
            "mdfs_raw_score": score,
            "runtime_seconds": elapsed
        })

        f.flush()

        done = idx + 1

        if done % 100 == 0 or done == len(manifest):
            print(
                f"[{done}/{len(manifest)}] "
                f"{filename} | "
                f"MDFS={score:.6f} | "
                f"{elapsed:.4f}s"
            )

total_elapsed = time.perf_counter() - start_all

final_df = pd.read_csv(output_path)

print("\n==============================")
print("FULL TID2013 EVALUATION COMPLETE")
print("==============================")
print("Rows:", len(final_df))
print("Unique filenames:", final_df["filename"].nunique())
print(
    "All scores finite:",
    final_df["mdfs_raw_score"]
        .map(math.isfinite)
        .all()
)
print(
    "Mean runtime/image:",
    final_df["runtime_seconds"].mean()
)
print(
    "Total wall time:",
    total_elapsed
)
