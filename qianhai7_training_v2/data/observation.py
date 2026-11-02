from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dl_fusion.data.regions import SquareGrid, aggregate_sparse_observations, build_square_grid


@dataclass(frozen=True)
class ObservationConfig:
    point_count_min: int = 50
    point_count_max: int = 100
    noise_std_mm: float = 0.2
    missing_probability: float = 0.05
    reference_mode: str = "first_step"
    interpolation_tolerance: float = 1e-3
    input_mode: str = "node"
    region_grid_size_m: float = 10.0
    min_points_per_region: int = 1
    coordinate_scale_to_m: float = 0.001

    def __post_init__(self) -> None:
        if self.input_mode not in ("node", "region"):
            raise ValueError("input_mode must be node or region")
        if self.region_grid_size_m <= 0:
            raise ValueError("region_grid_size_m must be positive")
        if self.min_points_per_region < 1:
            raise ValueError("min_points_per_region must be positive")
        if self.coordinate_scale_to_m <= 0:
            raise ValueError("coordinate_scale_to_m must be positive")


def build_observation_grid(
    node_coords: np.ndarray,
    deck_node_mask: np.ndarray,
    cell_size_m: float,
    coordinate_scale_to_m: float = 0.001,
) -> SquareGrid:
    """Build the shared, fixed grid from full deck geometry, never sparse samples."""
    coords = np.asarray(node_coords, dtype=np.float64)
    deck = np.asarray(deck_node_mask, dtype=bool).reshape(-1)
    if coords.ndim != 2 or coords.shape[0] != deck.size:
        raise ValueError("deck mask must align with node coordinates")
    return build_square_grid(coords[deck, :2] * float(coordinate_scale_to_m), cell_size_m)


def aggregate_observation_regions(
    observation: dict[str, np.ndarray],
    grid: SquareGrid,
    minimum_points: int,
    coordinate_scale_to_m: float = 0.001,
) -> dict[str, np.ndarray]:
    """Convert one sampled point view into fixed-grid observation tokens."""
    values, valid, quality, coverage = aggregate_sparse_observations(
        observation["values"],
        observation["valid_mask"],
        observation["quality"],
        np.asarray(observation["point_coords"][:, :2], dtype=np.float64)
        * float(coordinate_scale_to_m),
        grid,
        minimum_points=minimum_points,
    )
    return {
        **observation,
        "values": values,
        "valid_mask": valid,
        "quality": quality,
        "point_coords": grid.centers,
        "mapping_valid": grid.present_mask.copy(),
        "mapping_valid_mask": grid.present_mask.copy(),
        "mapping_distance": np.zeros(grid.region_count, dtype=np.float32),
        "point_present_mask": grid.present_mask.copy(),
        "coverage": coverage,
        "sampled_point_count": np.asarray(observation["values"].shape[1], dtype=np.int64),
        "point_count": np.asarray(grid.region_count, dtype=np.int64),
    }


@dataclass(frozen=True)
class ObservationMap:
    """Auditable H_space(S) mapping from points to four-node shell elements."""

    point_coords: np.ndarray
    node_indices: np.ndarray
    weights: np.ndarray
    mapping_valid: np.ndarray
    mapping_distance: np.ndarray

    def interpolate(self, nodal_values: np.ndarray) -> np.ndarray:
        values = np.asarray(nodal_values)
        if values.ndim == 1:
            return np.einsum("pk,pk->p", values[self.node_indices], self.weights)
        if values.ndim == 2:
            return np.einsum("tpk,pk->tp", values[:, self.node_indices], self.weights)
        raise ValueError(f"Expected nodal values with 1 or 2 dimensions, got {values.shape}")


def _quad_weights(xy: np.ndarray, point: np.ndarray, tolerance: float) -> tuple[np.ndarray, float] | None:
    """Solve the inverse bilinear map for an exported four-node quad."""
    p0, p1, p2, p3 = xy
    a = (p0 + p1 + p2 + p3) / 4.0
    b = (-p0 + p1 + p2 - p3) / 4.0
    c = (-p0 - p1 + p2 + p3) / 4.0
    d = (p0 - p1 + p2 - p3) / 4.0
    xi_eta = np.zeros(2, dtype=np.float64)
    for _ in range(12):
        xi, eta = xi_eta
        residual = a + b * xi + c * eta + d * xi * eta - point
        jacobian = np.column_stack((b + d * eta, c + d * xi))
        try:
            step = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            return None
        xi_eta -= step
        if np.linalg.norm(step) < 1e-8:
            break
    xi, eta = xi_eta
    error = float(np.linalg.norm(a + b * xi + c * eta + d * xi * eta - point))
    if error > tolerance or np.max(np.abs(xi_eta)) > 1.0 + 1e-6:
        return None
    weights = np.array(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)],
        dtype=np.float64,
    ) / 4.0
    return weights, error


def build_observation_map(
    point_coords: np.ndarray,
    node_coords: np.ndarray,
    node_ids: np.ndarray,
    deck_element_connectivity: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> ObservationMap:
    """Build a reproducible point-to-shell-element map.

    Points that do not map to a deck element are retained with `mapping_valid=False`
    and a nearest-node fallback. Their observations are masked by the caller.
    """
    coords = np.asarray(point_coords, dtype=np.float64)
    nodes = np.asarray(node_coords, dtype=np.float64)
    id_to_index = {int(node_id): index for index, node_id in enumerate(np.asarray(node_ids).tolist())}
    elements = np.asarray(deck_element_connectivity)
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError(f"Expected four-node deck elements, got {elements.shape}")
    element_nodes = np.array([[id_to_index[int(node_id)] for node_id in row] for row in elements], dtype=np.int64)
    element_xy = nodes[element_nodes][:, :, :2]
    mins = element_xy.min(axis=1)
    maxs = element_xy.max(axis=1)
    node_indices = np.empty((len(coords), 4), dtype=np.int64)
    weights = np.zeros((len(coords), 4), dtype=np.float64)
    mapping_valid = np.zeros(len(coords), dtype=bool)
    mapping_distance = np.full(len(coords), np.inf, dtype=np.float64)
    for point_index, point in enumerate(coords[:, :2]):
        candidate_indices = np.flatnonzero(np.all((point >= mins - tolerance) & (point <= maxs + tolerance), axis=1))
        best: tuple[float, np.ndarray] | None = None
        for element_index in candidate_indices.tolist():
            result = _quad_weights(element_xy[element_index], point, tolerance)
            if result is None:
                continue
            element_weights, error = result
            if best is None or error < best[0]:
                best = (error, element_weights)
                best_nodes = element_nodes[element_index]
        if best is not None:
            mapping_distance[point_index], weights[point_index] = best
            node_indices[point_index] = best_nodes
            mapping_valid[point_index] = True
        else:
            nearest = int(np.argmin(np.linalg.norm(nodes[:, :2] - point, axis=1)))
            node_indices[point_index] = nearest
            weights[point_index, 0] = 1.0
            mapping_distance[point_index] = float(np.linalg.norm(nodes[nearest, :2] - point))
    return ObservationMap(coords.astype(np.float32), node_indices, weights.astype(np.float32), mapping_valid, mapping_distance.astype(np.float32))


def sample_virtual_observation(
    node_coords: np.ndarray,
    deck_node_mask: np.ndarray,
    displacement_uz: np.ndarray,
    config: ObservationConfig,
    rng: np.random.Generator,
    *,
    node_ids: np.ndarray,
    deck_element_connectivity: np.ndarray,
    point_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample virtual coordinates and interpolate UZ through shell elements."""
    if config.point_count_min < 1 or config.point_count_max < config.point_count_min:
        raise ValueError("Invalid virtual point-count range")
    deck_indices = np.flatnonzero(deck_node_mask)
    if len(deck_indices) == 0:
        raise ValueError("The archive contains no deck nodes")
    count = (config.point_count_min if point_count is None else int(point_count))
    if count < config.point_count_min or count > config.point_count_max:
        raise ValueError("point_count must be within ObservationConfig bounds")
    replace = count > len(deck_indices)
    selected = rng.choice(deck_indices, size=count, replace=replace)
    point_coords = node_coords[selected]
    mapping = build_observation_map(
        point_coords,
        node_coords,
        node_ids,
        deck_element_connectivity,
        tolerance=config.interpolation_tolerance,
    )
    values = mapping.interpolate(displacement_uz).astype(np.float32)
    if config.reference_mode == "first_step":
        values = values - values[0:1]
    elif config.reference_mode != "none":
        raise ValueError(f"Unsupported reference mode: {config.reference_mode}")
    valid = rng.random(values.shape) >= config.missing_probability
    noise = rng.normal(0.0, config.noise_std_mm, size=values.shape).astype(np.float32)
    valid &= mapping.mapping_valid[None, :]
    observed = np.where(valid, values + noise, 0.0).astype(np.float32)
    quality = valid.astype(np.float32)
    return {
        "values": observed,
        "valid_mask": valid,
        "quality": quality,
        "point_coords": point_coords.astype(np.float32),
        "mapping_valid": mapping.mapping_valid,
        "mapping_distance": mapping.mapping_distance,
        "mapping_valid_mask": mapping.mapping_valid.copy(),
    }
