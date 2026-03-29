"""Visualize HED and RGB channel augmentations for H&E images."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Ruifrok and Johnston HED stain basis used by scikit-image.
RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float32,
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show HED and RGB channel augmentations for a folder of H&E images."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing H&E images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vis/he_augmentations_preview.png"),
        help="Path to the output comparison image.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Maximum number of images to visualize.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square resize applied before plotting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible augmentations.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    parser.add_argument(
        "--max-output-mb",
        type=float,
        default=10.0,
        help="Maximum output file size in MB.",
    )
    return parser.parse_args()


def list_images(input_dir: Path, limit: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {input_dir}")
    return image_paths[:limit]


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def rgb_to_hed(image: np.ndarray) -> np.ndarray:
    rgb = np.clip(image, 1e-6, 1.0)
    optical_density = -np.log(rgb)
    return optical_density @ HED_FROM_RGB.T


def hed_to_rgb(stains: np.ndarray) -> np.ndarray:
    optical_density = stains @ RGB_FROM_HED.T
    rgb = np.exp(-optical_density)
    return np.clip(rgb, 0.0, 1.0)


def apply_hed_augmentation(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: str,
) -> np.ndarray:
    stains = rgb_to_hed(image)

    strength_ranges = {
        "weak": {
            "scale": [(0.95, 1.05), (0.95, 1.05), (0.98, 1.02)],
            "shift": [(-0.03, 0.03), (-0.03, 0.03), (-0.01, 0.01)],
            "dab_weight": 0.15,
        },
        "medium": {
            "scale": [(0.88, 1.12), (0.88, 1.12), (0.96, 1.04)],
            "shift": [(-0.06, 0.06), (-0.06, 0.06), (-0.015, 0.015)],
            "dab_weight": 0.2,
        },
        "strong": {
            "scale": [(0.78, 1.22), (0.78, 1.22), (0.94, 1.06)],
            "shift": [(-0.12, 0.12), (-0.12, 0.12), (-0.02, 0.02)],
            "dab_weight": 0.25,
        },
    }
    if strength not in strength_ranges:
        raise ValueError(f"Unsupported HED strength: {strength}")

    profile = strength_ranges[strength]
    scale = np.array([rng.uniform(low, high) for low, high in profile["scale"]], dtype=np.float32)
    shift = np.array([rng.uniform(low, high) for low, high in profile["shift"]], dtype=np.float32)

    augmented = stains * scale + shift
    augmented[..., 2] *= profile["dab_weight"]
    return hed_to_rgb(augmented)


def permute_channels(image: np.ndarray, ordering: str) -> np.ndarray:
    lookup = {"R": 0, "G": 1, "B": 2}
    channel_indices = [lookup[channel] for channel in ordering]
    return image[..., channel_indices]


def build_figure(
    image_paths: list[Path],
    image_size: int,
    seed: int,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    rows = len(image_paths)
    fig, axes = plt.subplots(rows, 9, figsize=(28, 3.5 * rows), squeeze=False)

    column_titles = [
        "Original",
        "HED Weak",
        "HED Medium",
        "HED Strong",
        "RGB->RBG",
        "RGB->GBR",
        "RGB->GRB",
        "RGB->BGR",
        "RGB->BRG",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=18, pad=14)

    for row, path in enumerate(image_paths):
        image = load_image(path, image_size=image_size)
        panels = [
            image,
            apply_hed_augmentation(image, rng, strength="weak"),
            apply_hed_augmentation(image, rng, strength="medium"),
            apply_hed_augmentation(image, rng, strength="strong"),
            permute_channels(image, "RBG"),
            permute_channels(image, "GBR"),
            permute_channels(image, "GRB"),
            permute_channels(image, "BGR"),
            permute_channels(image, "BRG"),
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)
    return fig


def save_figure_with_size_limit(figure: plt.Figure, output_path: Path, max_output_mb: float) -> None:
    max_bytes = int(max_output_mb * 1024 * 1024)
    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        for quality in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50):
            buffer = io.BytesIO()
            figure.savefig(
                buffer,
                format="jpeg",
                dpi=200,
                bbox_inches="tight",
                pil_kwargs={"quality": quality},
            )
            if buffer.tell() <= max_bytes:
                output_path.write_bytes(buffer.getvalue())
                return
        output_path.write_bytes(buffer.getvalue())
        return

    if suffix == ".png":
        for dpi in (200, 180, 160, 140, 120, 100, 90, 80):
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
            if buffer.tell() <= max_bytes:
                output_path.write_bytes(buffer.getvalue())
                return
        output_path.write_bytes(buffer.getvalue())
        return

    raise ValueError(f"Unsupported output extension: {output_path.suffix}. Use .png, .jpg, or .jpeg")


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.input_dir, args.num_images)
    figure = build_figure(image_paths, image_size=args.image_size, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure_with_size_limit(figure, args.output, args.max_output_mb)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Saved augmentation preview to {args.output} ({size_mb:.2f} MB)")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
