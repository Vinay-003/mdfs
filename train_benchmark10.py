
import glob
import random
import time
import torch
from PIL import Image

from model import EffNet, APL, prepare_image


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Same models as authors' train.py
vgg = EffNet().to(device)
model = APL().to(device)


# Same training/reference folder
flist = glob.glob("./data/div500/*")

print("Total DIV500 files found:", len(flist))

assert len(flist) == 500, \
    f"Expected 500 reference images, found {len(flist)}"


# Fixed seed ONLY to make our benchmark image subset reproducible.
# It does not alter the real authors' train.py.
random.seed(0)
random.shuffle(flist)

benchmark_files = flist[:10]

print("Benchmark images:", len(benchmark_files))


# Synchronize before starting timer
if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.perf_counter()

feats_total = []

with torch.inference_mode():

    for i, f in enumerate(benchmark_files, start=1):

        image = Image.open(f).convert("RGB")

        x = prepare_image(image).to(device)

        feats_x = vgg(x)

        out, w = model(
            feats_x,
            select=1
        )

        feats_total.append(out)

        print(
            f"[{i:02d}/10] "
            f"{f.split('/')[-1]}"
        )


feats_total = torch.cat(
    feats_total,
    dim=0
)


if torch.cuda.is_available():
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start


# Save separately — DO NOT create/overwrite real MDFS_weights.pth
output = (
    "artifacts/"
    "MDFS_weights_10image_benchmark.pth"
)

torch.save(
    feats_total,
    output
)


mean_per_image = elapsed / len(benchmark_files)

estimated_500 = (
    mean_per_image * 500
)


print("\n==============================")
print("10-IMAGE BENCHMARK COMPLETE")
print("==============================")

print(
    f"Elapsed for 10 images: "
    f"{elapsed:.2f} seconds"
)

print(
    f"Mean per image: "
    f"{mean_per_image:.3f} seconds"
)

print(
    f"Estimated 500-image feature time: "
    f"{estimated_500:.2f} seconds"
)

print(
    f"Estimated 500-image feature time: "
    f"{estimated_500 / 60:.2f} minutes"
)

print(
    "Output tensor shape:",
    tuple(feats_total.shape)
)

print(
    "Finite:",
    torch.isfinite(feats_total)
         .all()
         .item()
)

print(
    "Temporary benchmark artifact:",
    output
)
