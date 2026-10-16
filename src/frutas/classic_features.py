from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import color, exposure
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureConfig:
    """Configuración reproducible del extractor clásico de características."""

    image_size: int = 64
    hsv_bins: int = 16
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    lbp_points: int = 24
    lbp_radius: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def color_feature_count(self) -> int:
        return 12 * self.hsv_bins + 60

    @property
    def lbp_feature_count(self) -> int:
        return 2 * (self.lbp_points + 2)


class BlockBalancedScaler(BaseEstimator, TransformerMixin):
    """Estandariza columnas y equilibra la varianza total entre color, HOG y LBP."""

    def __init__(self, color_dim: int = 252, lbp_dim: int = 52) -> None:
        self.color_dim = color_dim
        self.lbp_dim = lbp_dim

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "BlockBalancedScaler":
        self.scaler_ = StandardScaler().fit(x)
        total_dim = int(x.shape[1])
        c_dim = min(self.color_dim, total_dim)
        t_dim = min(self.lbp_dim, max(0, total_dim - c_dim))
        h_dim = max(0, total_dim - c_dim - t_dim)

        weights = np.ones(total_dim, dtype=np.float32)
        if c_dim > 0:
            weights[:c_dim] = 1.0 / np.sqrt(float(c_dim))
        if h_dim > 0:
            weights[c_dim : c_dim + h_dim] = 1.0 / np.sqrt(float(h_dim))
        if t_dim > 0:
            weights[c_dim + h_dim :] = 1.0 / np.sqrt(float(t_dim))
        self.weights_ = weights
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(x)
        return (scaled * self.weights_).astype(np.float32)


def safe_rgb(image: Image.Image) -> Image.Image:
    """Convierte una imagen PIL a RGB evitando advertencias de paleta transparente."""

    transposed = ImageOps.exif_transpose(image)
    if transposed.mode in ("P", "PA") and "transparency" in transposed.info:
        return transposed.convert("RGBA").convert("RGB")
    return transposed.convert("RGB")


def _normalized_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def _uniform_lbp_histogram(equalized_gray: np.ndarray, points: int, radius: int) -> np.ndarray:
    lbp = local_binary_pattern(
        img_as_ubyte(equalized_gray),
        points,
        radius,
        method="uniform",
    )
    lbp_bins = points + 2
    histogram, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def extract_classic_features(
    image: str | Path | Image.Image,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Extrae color (HSV + CIE Lab global/centro/espacial), forma (HOG) y textura (LBP)."""

    config = config or FeatureConfig()
    if isinstance(image, Image.Image):
        pil_image = safe_rgb(image)
    else:
        with Image.open(image) as opened:
            pil_image = safe_rgb(opened)

    size = config.image_size
    resized = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0

    hsv = color.rgb2hsv(rgb)
    lab = color.rgb2lab(rgb)
    lab_norm = np.stack(
        [
            np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0),
            np.clip((lab[:, :, 1] + 128.0) / 255.0, 0.0, 1.0),
            np.clip((lab[:, :, 2] + 128.0) / 255.0, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)

    q1 = size // 4
    q3 = size - q1
    mid = size // 2
    center_hsv = hsv[q1:q3, q1:q3, :]
    center_lab = lab_norm[q1:q3, q1:q3, :]

    color_histograms: list[np.ndarray] = []
    for channel in range(3):
        color_histograms.append(_normalized_histogram(hsv[:, :, channel], config.hsv_bins))
        color_histograms.append(_normalized_histogram(center_hsv[:, :, channel], config.hsv_bins))
        color_histograms.append(_normalized_histogram(lab_norm[:, :, channel], config.hsv_bins))
        color_histograms.append(_normalized_histogram(center_lab[:, :, channel], config.hsv_bins))

    regions = (
        (slice(0, mid), slice(0, mid)),
        (slice(0, mid), slice(mid, size)),
        (slice(mid, size), slice(0, mid)),
        (slice(mid, size), slice(mid, size)),
        (slice(q1, q3), slice(q1, q3)),
    )
    spatial_moments: list[np.ndarray] = []
    for row_slice, col_slice in regions:
        for color_space in (hsv, lab_norm):
            patch = color_space[row_slice, col_slice, :]
            spatial_moments.append(patch.mean(axis=(0, 1)).astype(np.float32))
            spatial_moments.append(patch.std(axis=(0, 1)).astype(np.float32))

    color_features = np.concatenate(color_histograms + spatial_moments).astype(np.float32)

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

    lbp_global = _uniform_lbp_histogram(equalized, config.lbp_points, config.lbp_radius)
    lbp_center = _uniform_lbp_histogram(
        equalized[q1:q3, q1:q3], config.lbp_points, config.lbp_radius
    )
    lbp_features = np.concatenate((lbp_global, lbp_center)).astype(np.float32)

    return np.concatenate((color_features, hog_features, lbp_features)).astype(np.float32)


