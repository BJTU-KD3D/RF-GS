"""Coarse-to-fine pruning used by RF-GS.

The coarse score follows LightGaussian: a Gaussian's accumulated rendering
importance is multiplied by a volume correction.  Pixel-GS already returns
the number of pixels to which each Gaussian contributes, so its accumulated
``pixels * opacity`` is equivalent to LightGaussian's CUDA importance counter.

The fine stage implements Eqs. (12)--(16) of RF-GS: KD-tree local density,
scale anisotropy, maximum-likelihood GEV fitting, and the conjunction of the
density and anisotropy outlier masks.
"""

from __future__ import print_function

import gc
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class RFGSPruningConfig:
    """Runtime configuration for RF-GS pruning."""

    score_prune_iteration: int = -1
    score_prune_percent: float = 0.10
    score_volume_power: float = 0.10
    score_camera_stride: int = 1
    fine_prune_iteration: int = 24_000
    kdtree_neighbors: int = 200
    density_std_ratio: float = 5.0
    gev_tail_probability: float = 0.05
    kdtree_chunk_size: int = 50_000
    kdtree_workers: int = -1
    reset_position_lr: bool = True


def add_rfgs_pruning_arguments(parser: Any) -> None:
    """Register RF-GS pruning options on an ``ArgumentParser``."""

    group = parser.add_argument_group("RF-GS pruning")
    group.add_argument(
        "--score_prune_iteration",
        type=int,
        default=-1,
        help=(
            "Iteration for the one-shot LightGaussian-style score pruning; "
            "a negative value selects the first coarse-pruning iteration and "
            "zero disables this stage."
        ),
    )
    group.add_argument(
        "--score_prune_percent",
        type=float,
        default=0.10,
        help="Fraction of the lowest-ranked Gaussians removed by score pruning.",
    )
    group.add_argument(
        "--score_volume_power",
        type=float,
        default=0.10,
        help="Volume exponent in the LightGaussian importance score.",
    )
    group.add_argument(
        "--score_camera_stride",
        type=int,
        default=1,
        help="Use every N-th training camera when accumulating importance.",
    )
    group.add_argument(
        "--fine_prune_iteration",
        type=int,
        default=24_000,
        help="Iteration for KD-tree/GEV fine pruning; zero disables this stage.",
    )
    group.add_argument(
        "--kdtree_neighbors",
        type=int,
        default=200,
        help="Number of non-self neighbors used for local-density estimation.",
    )
    group.add_argument(
        "--density_std_ratio",
        type=float,
        default=5.0,
        help="Kappa in the density threshold mean(d) + kappa * std(d).",
    )
    group.add_argument(
        "--gev_tail_probability",
        type=float,
        default=0.05,
        help="v in Eq. (16); the GEV threshold is the (1-v) quantile.",
    )
    group.add_argument(
        "--kdtree_chunk_size",
        type=int,
        default=50_000,
        help="Number of query points processed per KD-tree batch.",
    )
    group.add_argument(
        "--kdtree_workers",
        type=int,
        default=-1,
        help="Workers used by scipy.spatial.cKDTree.query (-1 uses all cores).",
    )
    group.add_argument(
        "--no_pruning_lr_reset",
        action="store_true",
        help="Do not restart the position learning-rate schedule after fine pruning.",
    )


def pruning_config_from_args(args: Any) -> RFGSPruningConfig:
    """Create a validated pruning configuration from parsed arguments."""

    config = RFGSPruningConfig(
        score_prune_iteration=args.score_prune_iteration,
        score_prune_percent=args.score_prune_percent,
        score_volume_power=args.score_volume_power,
        score_camera_stride=args.score_camera_stride,
        fine_prune_iteration=args.fine_prune_iteration,
        kdtree_neighbors=args.kdtree_neighbors,
        density_std_ratio=args.density_std_ratio,
        gev_tail_probability=args.gev_tail_probability,
        kdtree_chunk_size=args.kdtree_chunk_size,
        kdtree_workers=args.kdtree_workers,
        reset_position_lr=not args.no_pruning_lr_reset,
    )
    _validate_config(config)
    return config


def _validate_config(config: RFGSPruningConfig) -> None:
    if not 0.0 <= config.score_prune_percent < 1.0:
        raise ValueError("score_prune_percent must be in [0, 1).")
    if config.score_volume_power < 0.0:
        raise ValueError("score_volume_power must be non-negative.")
    if config.score_camera_stride < 1:
        raise ValueError("score_camera_stride must be at least 1.")
    if config.kdtree_neighbors < 1:
        raise ValueError("kdtree_neighbors must be at least 1.")
    if config.density_std_ratio < 0.0:
        raise ValueError("density_std_ratio must be non-negative.")
    if not 0.0 < config.gev_tail_probability < 1.0:
        raise ValueError("gev_tail_probability must be in (0, 1).")
    if config.kdtree_chunk_size < 1:
        raise ValueError("kdtree_chunk_size must be at least 1.")
    if config.kdtree_workers == 0:
        raise ValueError("kdtree_workers cannot be zero.")


def accumulate_render_importance(
    gaussians: Any,
    cameras: Sequence[Any],
    render_fn: Callable[..., Dict[str, torch.Tensor]],
    pipe: Any,
    background: torch.Tensor,
    depth_threshold: Optional[float] = None,
    camera_stride: int = 1,
) -> Tuple[torch.Tensor, int]:
    """Accumulate LightGaussian-compatible importance over training views.

    LightGaussian increments a Gaussian's score by its opacity for every pixel
    to which that Gaussian contributes.  Pixel-GS exposes the same contribution
    count as ``render_pkg["pixels"]``; multiplying the accumulated count by the
    activated opacity therefore produces the same score.
    """

    if camera_stride < 1:
        raise ValueError("camera_stride must be at least 1.")

    selected_cameras = list(cameras)[::camera_stride]
    if not selected_cameras:
        raise ValueError("Score pruning requires at least one training camera.")

    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity.detach().reshape(-1)
    point_count = int(xyz.shape[0])
    pixel_count = torch.zeros(point_count, dtype=opacity.dtype, device=xyz.device)

    with torch.no_grad():
        for camera in selected_cameras:
            render_kwargs = {}
            if depth_threshold is not None:
                render_kwargs["depth_threshold"] = depth_threshold
            render_pkg = render_fn(
                camera,
                gaussians,
                pipe,
                background,
                **render_kwargs
            )
            if "pixels" not in render_pkg:
                raise KeyError(
                    "Pixel-GS render output does not contain the per-Gaussian "
                    "'pixels' tensor required for score pruning."
                )
            pixels = render_pkg["pixels"].detach().reshape(-1)
            if pixels.numel() != point_count:
                raise ValueError(
                    "Per-Gaussian pixel count has length {}, expected {}.".format(
                        pixels.numel(), point_count
                    )
                )
            pixel_count.add_(pixels.to(device=xyz.device, dtype=opacity.dtype))

    return pixel_count * opacity, len(selected_cameras)


def lightgaussian_volume_score(
    gaussians: Any,
    render_importance: torch.Tensor,
    volume_power: float = 0.10,
) -> Tuple[torch.Tensor, float]:
    """Apply LightGaussian's Gaussian-volume correction to importance."""

    if volume_power < 0.0:
        raise ValueError("volume_power must be non-negative.")

    scaling = gaussians.get_scaling.detach()
    volume = torch.prod(scaling, dim=1).reshape(-1)
    importance = render_importance.detach().reshape(-1).to(
        device=volume.device, dtype=volume.dtype
    )
    if volume.numel() != importance.numel():
        raise ValueError("Volume and importance tensors must have equal lengths.")
    if volume.numel() == 0:
        return volume, 0.0

    # This is intentionally the same reference statistic used by LightGaussian.
    reference_index = min(int(volume.numel() * 0.9), volume.numel() - 1)
    sorted_volume = torch.sort(volume, descending=True).values
    eps = torch.finfo(volume.dtype).tiny
    reference_volume = sorted_volume[reference_index].clamp_min(eps)
    score = torch.pow(volume / reference_volume, volume_power) * importance

    finite = torch.isfinite(score)
    if not torch.all(finite):
        finite_values = score[finite]
        positive_fill = (
            finite_values.max()
            if finite_values.numel() > 0
            else torch.tensor(0.0, dtype=score.dtype, device=score.device)
        )
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isposinf(score), positive_fill, score)
        score = torch.where(torch.isneginf(score), torch.zeros_like(score), score)

    return score, float(reference_volume.item())


def score_based_coarse_prune(
    gaussians: Any,
    cameras: Sequence[Any],
    render_fn: Callable[..., Dict[str, torch.Tensor]],
    pipe: Any,
    background: torch.Tensor,
    prune_percent: float = 0.10,
    volume_power: float = 0.10,
    depth_threshold: Optional[float] = None,
    camera_stride: int = 1,
    min_remaining: int = 1,
) -> Dict[str, Any]:
    """Rank Gaussians by LightGaussian-style score and prune the lowest ones."""

    if not 0.0 <= prune_percent < 1.0:
        raise ValueError("prune_percent must be in [0, 1).")
    before = int(gaussians.get_xyz.shape[0])
    prune_count = min(int(math.floor(before * prune_percent)), before - min_remaining)
    if before == 0 or prune_count <= 0:
        return {
            "stage": "score",
            "before": before,
            "pruned": 0,
            "after": before,
            "cameras": 0,
        }

    started = time.time()
    render_importance, camera_count = accumulate_render_importance(
        gaussians=gaussians,
        cameras=cameras,
        render_fn=render_fn,
        pipe=pipe,
        background=background,
        depth_threshold=depth_threshold,
        camera_stride=camera_stride,
    )
    score, reference_volume = lightgaussian_volume_score(
        gaussians, render_importance, volume_power=volume_power
    )

    lowest_indices = torch.argsort(score)[:prune_count]
    prune_mask = torch.zeros(before, dtype=torch.bool, device=gaussians.get_xyz.device)
    prune_mask[lowest_indices] = True
    cutoff = float(score[lowest_indices[-1]].item()) if prune_count else float("nan")
    gaussians.prune_points(prune_mask)
    after = int(gaussians.get_xyz.shape[0])

    del render_importance, score, prune_mask, lowest_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "stage": "score",
        "before": before,
        "pruned": before - after,
        "after": after,
        "cameras": camera_count,
        "cutoff": cutoff,
        "reference_volume": reference_volume,
        "seconds": time.time() - started,
    }


def _query_average_neighbor_distance(
    points: np.ndarray,
    neighbor_count: int,
    chunk_size: int,
    workers: int,
) -> Tuple[np.ndarray, float]:
    """Build one global KD-tree and compute mean non-self neighbor distances."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "KD-tree fine pruning requires SciPy (pip install scipy)."
        ) from exc

    point_count = int(points.shape[0])
    if point_count < 2:
        return np.zeros(point_count, dtype=np.float64), 0.0
    if not np.all(np.isfinite(points)):
        raise ValueError("Gaussian coordinates contain NaN or infinity.")

    effective_neighbors = min(neighbor_count, point_count - 1)
    query_k = effective_neighbors + 1  # The nearest result is the point itself.
    build_started = time.time()
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    build_seconds = time.time() - build_started
    average_distance = np.empty(point_count, dtype=np.float64)

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        try:
            distances, _ = tree.query(
                points[start:end], k=query_k, workers=workers
            )
        except TypeError:
            # SciPy < 1.6 used n_jobs instead of workers.
            distances, _ = tree.query(
                points[start:end], k=query_k, n_jobs=workers
            )
        distances = np.asarray(distances)
        if distances.ndim == 1:
            distances = distances[:, None]
        average_distance[start:end] = np.mean(
            distances[:, 1:], axis=1, dtype=np.float64
        )

    del tree
    return average_distance, build_seconds


def fit_gev_anisotropy_threshold(
    anisotropy: np.ndarray,
    tail_probability: float = 0.05,
) -> Tuple[float, Dict[str, Any]]:
    """Fit a GEV distribution by MLE and evaluate RF-GS Eq. (16).

    SciPy uses ``c = -xi`` relative to the shape convention in the paper.
    Equation (16) is the GEV quantile at cumulative probability ``1 - v``.
    """

    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be in (0, 1).")
    finite_values = np.asarray(anisotropy, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("No finite anisotropy ratios are available for GEV fitting.")

    probability = 1.0 - tail_probability
    empirical_threshold = float(np.quantile(finite_values, probability))
    if finite_values.size < 3 or np.ptp(finite_values) <= np.finfo(np.float64).eps:
        return empirical_threshold, {
            "method": "empirical",
            "location": float(np.mean(finite_values)),
            "scale": 0.0,
            "shape_xi": 0.0,
        }

    try:
        from scipy.stats import genextreme
    except ImportError as exc:
        raise RuntimeError(
            "GEV fine pruning requires SciPy (pip install scipy)."
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scipy_shape, location, scale = genextreme.fit(finite_values)
            threshold = float(
                genextreme.ppf(
                    probability,
                    scipy_shape,
                    loc=location,
                    scale=scale,
                )
            )
        if not (
            np.isfinite(threshold)
            and np.isfinite(location)
            and np.isfinite(scale)
            and scale > 0.0
        ):
            raise ValueError("GEV fit returned non-finite parameters.")
        return threshold, {
            "method": "gev_mle",
            "location": float(location),
            "scale": float(scale),
            "shape_xi": float(-scipy_shape),
        }
    except (FloatingPointError, RuntimeError, ValueError):
        return empirical_threshold, {
            "method": "empirical_fallback",
            "location": float(np.mean(finite_values)),
            "scale": float(np.std(finite_values)),
            "shape_xi": 0.0,
        }


def kdtree_gev_fine_prune(
    gaussians: Any,
    k_neighbors: int = 200,
    density_std_ratio: float = 5.0,
    gev_tail_probability: float = 0.05,
    chunk_size: int = 50_000,
    workers: int = -1,
    epsilon: float = 1.0e-6,
    min_remaining: int = 1,
) -> Dict[str, Any]:
    """Prune points that are both spatially isolated and anisotropic."""

    if k_neighbors < 1:
        raise ValueError("k_neighbors must be at least 1.")
    if density_std_ratio < 0.0:
        raise ValueError("density_std_ratio must be non-negative.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    before = int(gaussians.get_xyz.shape[0])
    if before < 3:
        return {
            "stage": "fine",
            "before": before,
            "pruned": 0,
            "after": before,
            "reason": "fewer than three points",
        }

    started = time.time()
    points = (
        gaussians.get_xyz.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    scaling = (
        gaussians.get_scaling.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    average_distance, build_seconds = _query_average_neighbor_distance(
        points=points,
        neighbor_count=k_neighbors,
        chunk_size=chunk_size,
        workers=workers,
    )

    density_mean = float(np.mean(average_distance))
    density_std = float(np.std(average_distance))
    density_threshold = density_mean + density_std_ratio * density_std
    density_outlier = average_distance > density_threshold

    sorted_scaling = np.sort(scaling, axis=1)
    anisotropy = sorted_scaling[:, 2] / (sorted_scaling[:, 0] + epsilon)
    gev_started = time.time()
    anisotropy_threshold, gev_parameters = fit_gev_anisotropy_threshold(
        anisotropy, tail_probability=gev_tail_probability
    )
    gev_seconds = time.time() - gev_started
    anisotropy_outlier = anisotropy > anisotropy_threshold

    prune_mask_np = np.logical_and(density_outlier, anisotropy_outlier)
    requested_prune_count = int(np.count_nonzero(prune_mask_np))
    max_prune_count = max(0, before - min_remaining)
    if requested_prune_count > max_prune_count:
        candidate_indices = np.flatnonzero(prune_mask_np)
        # Keep only the most extreme candidates when the safety bound is reached.
        extremity = (
            average_distance[candidate_indices]
            / max(density_threshold, np.finfo(np.float64).tiny)
            * anisotropy[candidate_indices]
            / max(anisotropy_threshold, np.finfo(np.float64).tiny)
        )
        prune_mask_np[:] = False
        selected = candidate_indices[np.argsort(extremity)[-max_prune_count:]]
        prune_mask_np[selected] = True

    prune_mask = torch.from_numpy(prune_mask_np).to(
        device=gaussians.get_xyz.device, dtype=torch.bool
    )
    gaussians.prune_points(prune_mask)
    after = int(gaussians.get_xyz.shape[0])

    result = {
        "stage": "fine",
        "before": before,
        "pruned": before - after,
        "after": after,
        "density_outliers": int(np.count_nonzero(density_outlier)),
        "anisotropy_outliers": int(np.count_nonzero(anisotropy_outlier)),
        "density_mean": density_mean,
        "density_std": density_std,
        "density_threshold": float(density_threshold),
        "anisotropy_threshold": float(anisotropy_threshold),
        "gev_method": gev_parameters["method"],
        "gev_location": gev_parameters["location"],
        "gev_scale": gev_parameters["scale"],
        "gev_shape_xi": gev_parameters["shape_xi"],
        "kdtree_seconds": build_seconds,
        "gev_seconds": gev_seconds,
        "seconds": time.time() - started,
    }

    del (
        points,
        scaling,
        average_distance,
        sorted_scaling,
        anisotropy,
        density_outlier,
        anisotropy_outlier,
        prune_mask_np,
        prune_mask,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _print_pruning_result(iteration: int, result: Dict[str, Any]) -> None:
    if result["stage"] == "score":
        print(
            "[RF-GS][score pruning] iteration={} before={} pruned={} after={} "
            "cameras={} time={:.2f}s".format(
                iteration,
                result["before"],
                result["pruned"],
                result["after"],
                result.get("cameras", 0),
                result.get("seconds", 0.0),
            )
        )
    else:
        print(
            "[RF-GS][fine pruning] iteration={} before={} pruned={} after={} "
            "density_outliers={} anisotropy_outliers={} density_tau={:.6g} "
            "anisotropy_tau={:.6g} gev={} time={:.2f}s".format(
                iteration,
                result["before"],
                result["pruned"],
                result["after"],
                result.get("density_outliers", 0),
                result.get("anisotropy_outliers", 0),
                result.get("density_threshold", float("nan")),
                result.get("anisotropy_threshold", float("nan")),
                result.get("gev_method", "not-run"),
                result.get("seconds", 0.0),
            )
        )


class RFGSPruner:
    """Schedule score pruning, KD-tree/GEV pruning, and the LR restart."""

    def __init__(
        self,
        config: RFGSPruningConfig,
        densify_until_iteration: int,
        pruning_interval: int,
        resumed_iteration: int = 0,
    ) -> None:
        _validate_config(config)
        if pruning_interval < 1:
            raise ValueError("pruning_interval must be at least 1.")
        self.config = config
        if config.score_prune_iteration < 0:
            self.score_iteration = densify_until_iteration + pruning_interval
        else:
            self.score_iteration = config.score_prune_iteration
        self.fine_iteration = config.fine_prune_iteration
        self.score_done = (
            self.score_iteration == 0
            or config.score_prune_percent == 0.0
            or resumed_iteration >= self.score_iteration
        )
        self.fine_done = (
            self.fine_iteration <= 0 or resumed_iteration >= self.fine_iteration
        )
        self.lr_reset_iteration = (
            self.fine_iteration
            if self.fine_done and self.fine_iteration > 0
            else None
        )

    def learning_rate_iteration(self, iteration: int) -> int:
        """Return the scheduler step, accounting for a post-pruning restart."""

        if (
            self.config.reset_position_lr
            and self.lr_reset_iteration is not None
            and iteration > self.lr_reset_iteration
        ):
            return iteration - self.lr_reset_iteration
        return iteration

    def after_opacity_pruning(
        self,
        iteration: int,
        gaussians: Any,
        scene: Any,
        pipe: Any,
        background: torch.Tensor,
        render_fn: Callable[..., Dict[str, torch.Tensor]],
        depth_threshold: Optional[float],
    ) -> Sequence[Dict[str, Any]]:
        """Run RF-GS stages scheduled for this periodic pruning iteration."""

        results = []
        if not self.score_done and iteration >= self.score_iteration:
            result = score_based_coarse_prune(
                gaussians=gaussians,
                cameras=scene.getTrainCameras(),
                render_fn=render_fn,
                pipe=pipe,
                background=background,
                prune_percent=self.config.score_prune_percent,
                volume_power=self.config.score_volume_power,
                depth_threshold=depth_threshold,
                camera_stride=self.config.score_camera_stride,
            )
            self.score_done = True
            results.append(result)
            _print_pruning_result(iteration, result)

        if not self.fine_done and iteration >= self.fine_iteration:
            result = kdtree_gev_fine_prune(
                gaussians=gaussians,
                k_neighbors=self.config.kdtree_neighbors,
                density_std_ratio=self.config.density_std_ratio,
                gev_tail_probability=self.config.gev_tail_probability,
                chunk_size=self.config.kdtree_chunk_size,
                workers=self.config.kdtree_workers,
            )
            self.fine_done = True
            self.lr_reset_iteration = iteration
            if self.config.reset_position_lr:
                reset_lr = gaussians.update_learning_rate(0)
                result["reset_position_lr"] = reset_lr
                print(
                    "[RF-GS][fine pruning] reset position learning rate to {:.6g}".format(
                        reset_lr
                    )
                )
            results.append(result)
            _print_pruning_result(iteration, result)

        return results
