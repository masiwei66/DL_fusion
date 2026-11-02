"""Fixed metric square grids and quality-aware sparse-point aggregation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SquareGrid:
    """A dataset-fixed grid; empty geometric cells remain masked, never zero data."""

    origin_xy: tuple[float, float]
    cell_size_m: float
    rows: int
    columns: int
    present_mask: np.ndarray

    def __post_init__(self) -> None:
        present = np.asarray(self.present_mask, dtype=bool)
        if self.cell_size_m <= 0 or self.rows < 1 or self.columns < 1:
            raise ValueError("grid cell size and dimensions must be positive")
        if present.shape != (self.rows * self.columns,):
            raise ValueError("present_mask must have one entry per grid cell")
        if not present.any():
            raise ValueError("grid must contain at least one bridge-surface cell")

    @property
    def region_count(self) -> int:
        return int(self.rows * self.columns)

    @property
    def centers(self) -> np.ndarray:
        row, column = np.indices((self.rows, self.columns))
        x = self.origin_xy[0] + (column.reshape(-1) + 0.5) * self.cell_size_m
        y = self.origin_xy[1] + (row.reshape(-1) + 0.5) * self.cell_size_m
        return np.column_stack((x, y)).astype(np.float32)

    def assign(self, coordinates: np.ndarray) -> np.ndarray:
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("coordinates must have shape [P, >=2]")
        if not np.isfinite(points[:, :2]).all():
            raise ValueError("grid coordinates must be finite")
        column = np.floor((points[:, 0] - self.origin_xy[0]) / self.cell_size_m).astype(np.int64)
        row = np.floor((points[:, 1] - self.origin_xy[1]) / self.cell_size_m).astype(np.int64)
        upper_x = self.origin_xy[0] + self.columns * self.cell_size_m
        upper_y = self.origin_xy[1] + self.rows * self.cell_size_m
        on_upper_x = np.isclose(points[:, 0], upper_x, rtol=0.0, atol=1e-8)
        on_upper_y = np.isclose(points[:, 1], upper_y, rtol=0.0, atol=1e-8)
        column[on_upper_x] -= 1
        row[on_upper_y] -= 1
        in_grid = (row >= 0) & (row < self.rows) & (column >= 0) & (column < self.columns)
        result = np.full(len(points), -1, dtype=np.int64)
        cell = row[in_grid] * self.columns + column[in_grid]
        result[in_grid] = np.where(self.present_mask[cell], cell, -1)
        return result


def build_square_grid(
    surface_coordinates: np.ndarray,
    cell_size_m: float,
    *,
    origin_xy: tuple[float, float] | None = None,
) -> SquareGrid:
    """Build once from full bridge geometry; never infer bounds from sampled points."""
    coords = np.asarray(surface_coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        raise ValueError("surface_coordinates must be a non-empty [N, >=2] array")
    xy = coords[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError("surface coordinates must be finite metric coordinates")
    size = float(cell_size_m)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("cell_size_m must be a positive finite value")
    origin = (
        (float(np.floor(xy[:, 0].min() / size) * size), float(np.floor(xy[:, 1].min() / size) * size))
        if origin_xy is None
        else (float(origin_xy[0]), float(origin_xy[1]))
    )
    maximum = xy.max(axis=0)
    columns = max(1, int(np.ceil((maximum[0] - origin[0]) / size)))
    rows = max(1, int(np.ceil((maximum[1] - origin[1]) / size)))
    cell_count = rows * columns
    if cell_count > 1_000_000:
        raise ValueError(
            f"Grid would contain {cell_count:,} cells; check that coordinates are "
            "converted to meters before applying a metric cell size"
        )
    col = np.floor((xy[:, 0] - origin[0]) / size).astype(np.int64)
    row = np.floor((xy[:, 1] - origin[1]) / size).astype(np.int64)
    col[col == columns] = columns - 1
    row[row == rows] = rows - 1
    present = np.zeros(cell_count, dtype=bool)
    present[row * columns + col] = True
    # Areas enclosed by surface node coordinates can include cells without a node
    # in sparse meshes. Keep only cells with a geometric support point.
    return SquareGrid(origin, size, rows, columns, present)


def aggregate_sparse_observations(
    values: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    coordinates: np.ndarray,
    grid: SquareGrid,
    *,
    minimum_points: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool sparse points to grid cells using valid quality-weighted means.

    Returns ``values, valid_mask, quality, coverage`` with shapes ``[T, R]``.
    Coverage is the fraction of the sampled points assigned to a cell that remain
    valid at each acquisition; geometric empty cells stay represented only by
    ``grid.present_mask`` and cannot be mistaken for observed zeroes.
    """
    source = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    weights = np.asarray(quality, dtype=np.float32)
    # Keep metric coordinates in float64: rounding a value near a cell edge in
    # float32 can move it into the next region and inflate grid dimensions.
    points = np.asarray(coordinates, dtype=np.float64)
    if source.ndim != 2 or valid.shape != source.shape or weights.shape != source.shape:
        raise ValueError("values, valid_mask and quality must share [T, P] shape")
    if points.shape != (source.shape[1], 2):
        raise ValueError("coordinates must have one [x,y] pair per sparse point")
    if minimum_points < 1:
        raise ValueError("minimum_points must be positive")
    cell_ids = grid.assign(points)
    valid &= np.isfinite(source) & np.isfinite(weights) & (weights > 0)
    n_time, n_points = source.shape
    count = np.zeros((n_time, grid.region_count), dtype=np.int32)
    denom = np.zeros((n_time, grid.region_count), dtype=np.float64)
    numerator = np.zeros_like(denom)
    quality_sum = np.zeros_like(denom)
    sampled_count = np.bincount(cell_ids[cell_ids >= 0], minlength=grid.region_count)
    for point_index, cell_id in enumerate(cell_ids.tolist()):
        if cell_id < 0:
            continue
        keep = valid[:, point_index]
        point_weight = np.where(keep, weights[:, point_index], 0.0).astype(np.float64)
        count[:, cell_id] += keep.astype(np.int32)
        denom[:, cell_id] += point_weight
        numerator[:, cell_id] += source[:, point_index] * point_weight
        quality_sum[:, cell_id] += point_weight
    out_valid = (count >= int(minimum_points)) & (denom > 0) & grid.present_mask[None, :]
    out_values = np.zeros((n_time, grid.region_count), dtype=np.float32)
    np.divide(numerator, denom, out=out_values, where=out_valid)
    out_quality = np.zeros_like(out_values)
    np.divide(quality_sum, np.maximum(count, 1), out=out_quality, where=out_valid)
    coverage = np.zeros_like(out_values)
    expected = np.maximum(sampled_count, 1)[None, :]
    np.divide(count, expected, out=coverage, where=grid.present_mask[None, :])
    out_quality[~out_valid] = 0.0
    coverage[:, ~grid.present_mask] = 0.0
    return out_values, out_valid, out_quality, coverage
