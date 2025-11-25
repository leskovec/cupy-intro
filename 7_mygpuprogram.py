# ----------------------------------------
# SVD-based image filtering using CuPy (GPU)
# ----------------------------------------
# - Loads a JPEG
# - Moves it to GPU with CuPy
# - Applies low-rank SVD approximation (per color channel)
# - Saves the filtered image as JPEG
#

import argparse
import time

import numpy as np
import cupy as cp
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVD-based JPEG filtering with CuPy"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="wileecoyote.jpg",  # you can change this
        help="Input JPEG image path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="wileecoyote_svd.jpg",
        help="Output JPEG image path.",
    )
    parser.add_argument(
        "--rank",
        "-k",
        type=int,
        default=50,
        help="Rank for SVD approximation.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert to grayscale before SVD.",
    )
    return parser.parse_args()


def load_image_cpu(path, grayscale=False):
    """Load image on CPU as float32 array in [0,1]."""
    img = Image.open(path).convert("L" if grayscale else "RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, img.mode


def save_image_cpu(arr_float, mode, path):
    """Save float32 array in [0,1] to disk as uint8 image."""
    arr_clipped = np.clip(arr_float, 0.0, 1.0)
    arr_uint8 = (arr_clipped * 255.0).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode=mode)
    img.save(path, format="JPEG")


def svd_low_rank_gpu(channel_gpu, k):
    """
    Compute low-rank approximation of a 2D image channel on GPU using SVD.

    channel_gpu: 2D CuPy array (H, W)
    k: rank
    """
    H, W = channel_gpu.shape
    k = min(k, H, W)  # safety

    # SVD on GPU
    # U: (H, H)
    # S: (min(H,W))
    # Vt: (W, W)
    # We explicitly ask for reduced form to avoid wasting work/memory.
    U, S, Vt = cp.linalg.svd(channel_gpu, full_matrices=True)  
    # Truncate to rank k
    U_k = U[:, :k]                             # (H, k)
    S_k = S[:k]                                # (k,)
    Vt_k = Vt[:k, :]                           # (k, W)

    # Reconstruct: U_k * diag(S_k) * Vt_k
    # Use broadcasting instead of explicit diag
    US = U_k * S_k[cp.newaxis, :]              # (H, k)
    recon = US @ Vt_k                          # (H, W)

    return recon


def main():
    args = parse_args()

    print("=== SVD Image Filtering with CuPy ===")
    print(f"Input image : {args.input}")
    print(f"Output image: {args.output}")
    print(f"Rank k      : {args.rank}")
    print(f"Grayscale   : {args.grayscale}")

    # ----------------------------------------------------------------------
    # Load image on CPU
    # ----------------------------------------------------------------------
    t0 = time.time()
    img_cpu, mode = load_image_cpu(args.input, grayscale=args.grayscale)
    t_load = time.time() - t0
    print(f"Loaded image on CPU: shape={img_cpu.shape}, mode={mode}, time={t_load:.3f}s")

    # ----------------------------------------------------------------------
    # Move to GPU (explicit transfer)
    # ----------------------------------------------------------------------
    t1 = time.time()
    img_gpu = cp.asarray(img_cpu)  # host -> device
    cp.cuda.runtime.deviceSynchronize()
    t_htod = time.time() - t1
    print(f"Transferred image CPU -> GPU in {t_htod:.3f}s")

    # ----------------------------------------------------------------------
    # SVD per channel
    # ----------------------------------------------------------------------
    t2 = time.time()
    if mode == "L":
        # Grayscale: single channel, shape (H, W)
        recon_gpu = svd_low_rank_gpu(img_gpu, args.rank)
    else:
        # Color (RGB): shape (H, W, 3)
        H, W, C = img_gpu.shape
        assert C == 3, "Only RGB images are supported."

        # We'll loop over channels on GPU; C=3 so loop overhead is negligible.
        recon_channels = []
        for c in range(C):
            channel = img_gpu[:, :, c]
            recon_c = svd_low_rank_gpu(channel, args.rank)
            recon_channels.append(recon_c[..., cp.newaxis])

        recon_gpu = cp.concatenate(recon_channels, axis=2)  # (H, W, 3)

    cp.cuda.runtime.deviceSynchronize()
    t_svd = time.time() - t2
    print(f"SVD + reconstruction on GPU took {t_svd:.3f}s")

    # ----------------------------------------------------------------------
    # Move result back to CPU
    # ----------------------------------------------------------------------
    t3 = time.time()
    recon_cpu = cp.asnumpy(recon_gpu)  # device -> host
    t_dtoh = time.time() - t3
    print(f"Transferred result GPU -> CPU in {t_dtoh:.3f}s")

    # ----------------------------------------------------------------------
    # Save as JPEG
    # ----------------------------------------------------------------------
    t4 = time.time()
    save_image_cpu(recon_cpu, mode, args.output)
    t_save = time.time() - t4
    print(f"Saved filtered image to {args.output} in {t_save:.3f}s")

    total = time.time() - t0
    print(f"Total runtime: {total:.3f}s")
    print("Done.")


if __name__ == "__main__":
    main()