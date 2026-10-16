from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, exposure
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte


@dataclass(frozen=True)
class FeatureConfig:
    """Configuración reproducible del baseline clásico."""

    image_size: int = 64
    hsv_bins: int = 16
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    lbp_points: int = 24
    lbp_radius: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def extract_classic_features(
    image: str | Path | Image.Image,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Extrae color HSV, forma HOG y textura LBP en un vector float32."""

    config = config or FeatureConfig()
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")

    resized = pil_image.resize(
        (config.image_size, config.image_size),
        Image.Resampling.LANCZOS,
    )
    rgb = np.asarray(resized, dtype=np.float32) / 255.0

    hsv = color.rgb2hsv(rgb)
    hsv_features = np.concatenate(
        [_normalized_histogram(hsv[:, :, channel], config.hsv_bins) for channel in range(3)]
    )

    grayscale = color.rgb2gray(rgb)
    equalized = exposure.equalize_hist(grayscale)
    hog_features = hog(
        equalized,
        orientations=config.hog_orientations,
        pixels_per_cell=config.hog_pixels_per_cell,
        cells_per_block=config.hog_cells_per_block,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    ).astype(np.float32)

    lbp = local_binary_pattern(
        img_as_ubyte(equalized),
        config.lbp_points,
        config.lbp_radius,
        method="uniform",
    )
    lbp_bins = config.lbp_points + 2
    lbp_features, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    lbp_features = lbp_features.astype(np.float32)
    lbp_features /= max(float(lbp_features.sum()), 1.0)

    return np.concatenate((hsv_features, hog_features, lbp_features)).astype(np.float32)

