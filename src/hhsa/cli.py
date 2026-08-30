from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import convolve2d

from .core import decompose
from .spectrum import project


def main() -> None:
    p = argparse.ArgumentParser(description="Python port of neurohhsa_ex1.m")
    p.add_argument("--input", type=Path, default=Path("data/hhsa_ex1_data.mat"))
    p.add_argument("--output", type=Path, default=Path("outputs/HHS_ex1_python.png"))
    p.add_argument("--arrays", type=Path, default=Path("outputs/HHS_ex1_python.mat"))
    p.add_argument("--max-imfs", type=int, default=-1)
    p.add_argument("--max-modulation-imfs", type=int, default=-1)
    args = p.parse_args()
    signal = loadmat(args.input)["data"].squeeze()
    result = decompose(signal, 1000, max_imfs=args.max_imfs,
                       max_modulation_imfs=args.max_modulation_imfs)
    spectrum = project(result)
    total = spectrum.power.sum(axis=3).sum(axis=2)
    kernel = np.array([[.0276818087794658, .111014893010991, .0276818087794658],
                       [.111014893010991, .445213192838173, .111014893010991],
                       [.0276818087794658, .111014893010991, .0276818087794658]])
    smooth = convolve2d(convolve2d(total.T, kernel, mode="same"), kernel, mode="same")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.arrays, {"IMF": result.IMF, "IMF2": result.IMF2, "fm": result.fm,
                          "am": result.am, "FM": result.FM, "AM": result.AM,
                          "All_nt": spectrum.power})
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(np.log(np.maximum(smooth, np.finfo(float).tiny)), origin="lower",
                      extent=(0, 7, -1, 6), aspect="auto", vmin=-8, vmax=4)
    ax.plot([0, 6], [0, 6], "k:", linewidth=2)
    ax.set_xlabel("carrier frequency (Hz)")
    ax.set_ylabel("AM frequency (Hz)")
    ticks = np.arange(0, 8)
    ax.set_xticks(ticks, [str(2**i) for i in ticks])
    yticks = np.arange(-1, 7)
    ax.set_yticks(yticks, [f"{2.0**i:g}" for i in yticks])
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output} and {args.arrays}")


if __name__ == "__main__":
    main()
