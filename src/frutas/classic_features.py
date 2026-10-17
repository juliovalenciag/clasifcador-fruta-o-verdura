from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import color, exposure, filters
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureConfig:
    """Configuración reproducible del extractor clásico multiescala."""

    image_size: int = 64
    hsv_bins: int = 16
    index_bins: int = 12
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    lbp_points: int = 24
    lbp_radius: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def color_feature_count(self) -> int:
        return 14 * self.hsv_bins + 8 * self.index_bins + 519

    @property
    def hog_feature_count(self) -> int:
        return 44 + 46 * self.hog_orientations

    @property
    def lbp_feature_count(self) -> int:
        return 56 + 2 * (self.lbp_points + 2)

    @property
    def total_feature_count(self) -> int:
        return (
            self.color_feature_count
            + self.hog_feature_count
            + self.lbp_feature_count
        )


class BlockBalancedScaler(BaseEstimator, TransformerMixin):
    """Estandariza columnas y equilibra la varianza total entre color, HOG y LBP."""

    def __init__(self, color_dim: int = 839, lbp_dim: int = 108) -> None:
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


class AdaptivePCA(BaseEstimator, TransformerMixin):
    """Proyección ortogonal PCA que adapta automáticamente n_components al tamaño muestral."""

    def __init__(self, max_components: int = 256, random_state: int = 42) -> None:
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "AdaptivePCA":
        n_samples, n_features = x.shape
        n_comp = min(int(self.max_components), max(1, n_samples - 1), n_features)
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state).fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca_.transform(x).astype(np.float32)


def safe_rgb(image: Image.Image) -> Image.Image:
    """Convierte una imagen PIL a RGB evitando advertencias de paleta transparente."""

    transposed = ImageOps.exif_transpose(image)
    if transposed.mode in ("P", "PA") and "transparency" in transposed.info:
        return transposed.convert("RGBA").convert("RGB")
    return transposed.convert("RGB")


@lru_cache(maxsize=8)
def _radial_ring_masks(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) * 0.5
    dist = np.hypot(yy - center, xx - center) / max(size * 0.5, 1.0)
    inner = dist <= 0.42
    mid = (dist > 0.42) & (dist <= 0.78)
    outer = dist > 0.78
    return inner, mid, outer


def _fast_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    indices = np.clip((values.ravel() * bins).astype(np.int32), 0, bins - 1)
    histogram = np.bincount(indices, minlength=bins).astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def _weighted_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> np.ndarray:
    indices = np.clip((values.ravel() * bins).astype(np.int32), 0, bins - 1)
    histogram = np.bincount(
        indices, weights=weights.ravel(), minlength=bins
    ).astype(np.float32)
    return histogram / max(float(histogram.sum()), 1e-5)


def _uniform_lbp_pair(
    ubyte_equalized: np.ndarray,
    q1: int,
    q3: int,
    points: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    bins = points + 2
    lbp_full = local_binary_pattern(
        ubyte_equalized, points, radius, method="uniform"
    ).astype(np.int32)
    global_hist = np.bincount(
        np.clip(lbp_full.ravel(), 0, bins - 1), minlength=bins
    ).astype(np.float32)
    global_hist /= max(float(global_hist.sum()), 1.0)

    center_patch = lbp_full[q1:q3, q1:q3]
    center_hist = np.bincount(
        np.clip(center_patch.ravel(), 0, bins - 1), minlength=bins
    ).astype(np.float32)
    center_hist /= max(float(center_hist.sum()), 1.0)
    return global_hist, center_hist


def extract_classic_features(
    image: str | Path | Image.Image,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Extrae descriptores clásicos de cromaticidad 1D/2D, índices botánicos, figura-fondo, bordes, HOG multiescala y LBP multirradio."""

    config = config or FeatureConfig()
    if isinstance(image, Image.Image):
        pil_image = safe_rgb(image)
    else:
        with Image.open(image) as opened:
            pil_image = safe_rgb(opened)

    size = max(32, int(config.image_size))
    resized = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    hsv = color.rgb2hsv(rgb).astype(np.float32)
    lab = color.rgb2lab(rgb).astype(np.float32)
    l_norm = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)
    a_norm = np.clip((lab[:, :, 1] + 128.0) / 255.0, 0.0, 1.0)
    b_norm = np.clip((lab[:, :, 2] + 128.0) / 255.0, 0.0, 1.0)
    chroma = np.clip(np.hypot(lab[:, :, 1], lab[:, :, 2]) / 128.0, 0.0, 1.0)
    lab_norm = np.stack((l_norm, a_norm, b_norm), axis=-1).astype(np.float32)

    exg = np.clip((2.0 * green - red - blue + 1.0) * 0.5, 0.0, 1.0)
    rg_diff = np.clip((red - green + 1.0) * 0.5, 0.0, 1.0)
    yb_diff = np.clip(((red + green) * 0.5 - blue + 1.0) * 0.5, 0.0, 1.0)
    indices_map = np.stack((exg, rg_diff, yb_diff, chroma), axis=-1).astype(np.float32)

    q1 = size // 4
    q3 = size - q1
    center_hsv = hsv[q1:q3, q1:q3, :]
    center_lab = lab_norm[q1:q3, q1:q3, :]
    center_ind = indices_map[q1:q3, q1:q3, :]

    color_blocks: list[np.ndarray] = []
    for channel in range(3):
        color_blocks.append(_fast_histogram(hsv[:, :, channel], config.hsv_bins))
        color_blocks.append(_fast_histogram(center_hsv[:, :, channel], config.hsv_bins))
        color_blocks.append(_fast_histogram(lab_norm[:, :, channel], config.hsv_bins))
        color_blocks.append(_fast_histogram(center_lab[:, :, channel], config.hsv_bins))

    for channel in range(4):
        color_blocks.append(
            _fast_histogram(indices_map[:, :, channel], config.index_bins)
        )
        color_blocks.append(
            _fast_histogram(center_ind[:, :, channel], config.index_bins)
        )

    color_blocks.append(
        _weighted_histogram(hsv[:, :, 0], hsv[:, :, 1], config.hsv_bins)
    )
    color_blocks.append(
        _weighted_histogram(center_hsv[:, :, 0], center_hsv[:, :, 1], config.hsv_bins)
    )

    for h_patch, s_patch in (
        (hsv[:, :, 0], hsv[:, :, 1]),
        (center_hsv[:, :, 0], center_hsv[:, :, 1]),
    ):
        h_idx = np.clip((h_patch.ravel() * 8).astype(np.int32), 0, 7)
        s_idx = np.clip((s_patch.ravel() * 4).astype(np.int32), 0, 3)
        hs2d = np.bincount(h_idx * 4 + s_idx, minlength=32).astype(np.float32)
        color_blocks.append(hs2d / max(float(hs2d.sum()), 1.0))

    for a_patch, b_patch in (
        (lab_norm[:, :, 1], lab_norm[:, :, 2]),
        (center_lab[:, :, 1], center_lab[:, :, 2]),
    ):
        a_idx = np.clip(((a_patch.ravel() - 0.2) / 0.6 * 6).astype(np.int32), 0, 5)
        b_idx = np.clip(((b_patch.ravel() - 0.2) / 0.6 * 6).astype(np.int32), 0, 5)
        ab2d = np.bincount(a_idx * 6 + b_idx, minlength=36).astype(np.float32)
        color_blocks.append(ab2d / max(float(ab2d.sum()), 1.0))

    ring_inner, ring_mid, ring_outer = _radial_ring_masks(size)
    all_channels = np.concatenate((rgb, hsv, lab_norm, indices_map), axis=-1)
    ring_means: list[np.ndarray] = []
    for mask in (ring_inner, ring_mid, ring_outer):
        pixels = all_channels[mask]
        mean_vec = pixels.mean(axis=0).astype(np.float32)
        std_vec = pixels.std(axis=0).astype(np.float32)
        color_blocks.extend((mean_vec, std_vec))
        ring_means.append(mean_vec)
    color_blocks.append((ring_means[0] - ring_means[2]).astype(np.float32))
    color_blocks.append((ring_means[1] - ring_means[2]).astype(np.float32))

    border_signature = all_channels[ring_outer, 6:10].mean(axis=0)
    fg_dist = np.linalg.norm(all_channels[:, :, 6:10] - border_signature, axis=-1)
    fg_threshold = max(float(np.median(fg_dist)), 0.04)
    fg_mask = fg_dist >= fg_threshold
    if int(fg_mask.sum()) >= 16:
        fg_pixels = all_channels[fg_mask]
        bg_pixels = (
            all_channels[~fg_mask]
            if int((~fg_mask).sum()) >= 8
            else all_channels[ring_outer]
        )
        fg_mean = fg_pixels.mean(axis=0).astype(np.float32)
        fg_std = fg_pixels.std(axis=0).astype(np.float32)
        bg_mean = bg_pixels.mean(axis=0).astype(np.float32)
        fg_y, fg_x = np.nonzero(fg_mask)
        area_ratio = float(len(fg_y)) / float(size * size)
        cy = float(fg_y.mean()) / float(size)
        cx = float(fg_x.mean()) / float(size)
        sy = float(fg_y.std()) / float(size) + 1e-5
        sx = float(fg_x.std()) / float(size) + 1e-5
        aspect = float(np.clip(sy / sx, 0.2, 5.0) / 5.0)
        color_blocks.extend(
            (
                fg_mean,
                fg_std,
                (fg_mean - bg_mean).astype(np.float32),
                np.array([area_ratio, cy, cx, sy, sx, aspect], dtype=np.float32),
            )
        )
    else:
        color_blocks.append(np.zeros(45, dtype=np.float32))

    row_splits = np.array_split(np.arange(size), 3)
    col_splits = np.array_split(np.arange(size), 3)
    for r_idx in row_splits:
        for c_idx in col_splits:
            patch = all_channels[np.ix_(r_idx, c_idx)]
            color_blocks.append(patch.mean(axis=(0, 1)).astype(np.float32))
            color_blocks.append(patch.std(axis=(0, 1)).astype(np.float32))

    color_features = np.concatenate(color_blocks).astype(np.float32)

    grayscale = color.rgb2gray(rgb).astype(np.float32)
    equalized = exposure.equalize_hist(grayscale).astype(np.float32)
    sobel_mag = filters.sobel(grayscale).astype(np.float32)
    laplace_mag = np.abs(filters.laplace(grayscale)).astype(np.float32)

    shape_blocks: list[np.ndarray] = []
    grid_rows = np.array_split(np.arange(size), 4)
    grid_cols = np.array_split(np.arange(size), 4)
    for edge_map in (sobel_mag, laplace_mag):
        for mask in (ring_inner, ring_mid, ring_outer):
            vals = edge_map[mask]
            shape_blocks.append(
                np.array([vals.mean(), vals.std()], dtype=np.float32)
            )
        grid_vals = np.array(
            [
                [edge_map[np.ix_(gr, gc)].mean() for gc in grid_cols]
                for gr in grid_rows
            ],
            dtype=np.float32,
        ).ravel()
        shape_blocks.append(grid_vals)

    hog_surface = (
        equalized
        if size == 64
        else np.asarray(
            Image.fromarray(img_as_ubyte(equalized)).resize(
                (64, 64), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    hog_raw = hog(
        hog_surface,
        orientations=config.hog_orientations,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=False,
    ).astype(np.float32)
    hog_cell = hog_raw.mean(axis=(2, 3))
    macro_slices = (slice(0, 3), slice(2, 5), slice(4, 7))
    pooled_mean = np.array(
        [
            [hog_cell[sy, sx].mean(axis=(0, 1)) for sx in macro_slices]
            for sy in macro_slices
        ],
        dtype=np.float32,
    ).ravel()
    pooled_max = np.array(
        [
            [hog_cell[sy, sx].max(axis=(0, 1)) for sx in macro_slices]
            for sy in macro_slices
        ],
        dtype=np.float32,
    ).ravel()
    shape_blocks.extend(
        (
            pooled_mean,
            pooled_max,
            hog_cell[1:6, 1:6].ravel(),
            hog_cell.mean(axis=(0, 1)),
            hog_cell.std(axis=(0, 1)),
            hog_cell.max(axis=(0, 1)),
        )
    )
    hog_features = np.concatenate(shape_blocks).astype(np.float32)

    ubyte_eq = img_as_ubyte(equalized)
    lbp_blocks: list[np.ndarray] = []
    for pts, rad in ((8, 1), (16, 2), (config.lbp_points, config.lbp_radius)):
        g_hist, c_hist = _uniform_lbp_pair(ubyte_eq, q1, q3, pts, rad)
        lbp_blocks.extend((g_hist, c_hist))
    lbp_features = np.concatenate(lbp_blocks).astype(np.float32)

    return np.concatenate((color_features, hog_features, lbp_features)).astype(
        np.float32
    )
