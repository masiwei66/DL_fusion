"""Dataset loader for multi-condition bridge safety assessment.

The loader accepts compact ``result_*.json`` metadata plus HDF5 response arrays.
It also keeps backward compatibility with the older flat JSON sample format.
"""

import glob
import hashlib
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .DL_config import (
        CANDIDATE_IDS,
        DYNAMIC_NODES,
        REGION_NAMES,
        STATIC_NODES,
        SUPPORT_NODES,
    )
    from .safety_rules import build_safety_labels
except ImportError:  # Support running scripts directly from this folder.
    from DL_config import (
        CANDIDATE_IDS,
        DYNAMIC_NODES,
        REGION_NAMES,
        STATIC_NODES,
        SUPPORT_NODES,
    )
    from safety_rules import build_safety_labels


def _numeric_keys(mapping):
    return [int(k) for k in mapping if str(k).lstrip("-").isdigit()]


def _ordered_nodes_from_map(node_map):
    return [
        int(node)
        for node, _idx in sorted(
            node_map.items(), key=lambda item: int(item[1])
        )
    ]


def _response_map(sample, *keys):
    maps = sample.get("node_maps", {}) or {}
    for key in keys:
        if key in maps:
            return maps[key]
    return None


def _nested_response(sample, response_key):
    """Return a response from either the legacy flat or V2 nested schema."""
    if response_key in sample:
        return sample[response_key]
    paths = {
        "disp": ("responses", "static", "disp"),
        "strain": ("responses", "static", "strain"),
        "ace": ("responses", "dynamic", "ace"),
        "time_s": ("responses", "dynamic", "time_s"),
        "force_N": ("responses", "dynamic", "force_N"),
    }
    current = sample
    for key in paths.get(response_key, ()):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _nodes_from_sample(sample, response_key, fallback_nodes, *map_keys):
    node_map = _response_map(sample, *(map_keys or (response_key,)))
    if node_map:
        return _ordered_nodes_from_map(node_map)
    if fallback_nodes:
        return [int(n) for n in fallback_nodes]
    response = _nested_response(sample, response_key)
    if response is None:
        return []
    return list(range(np.asarray(response).shape[1]))


def _material_ids_from_sample(sample):
    if "material_ids" in sample:
        return [int(v) for v in sample["material_ids"]]
    material_data = sample.get("scaling_factors", {}).get("material", {})
    return sorted(_numeric_keys(material_data))


def _material_scaling_from_sample(sample, material_ids):
    if "material_scaling_factors" in sample:
        values = np.asarray(sample["material_scaling_factors"], dtype=np.float32)
        if len(values) == len(material_ids):
            return values

    material_data = sample.get("scaling_factors", {}).get("material", {})
    values = []
    for mid in material_ids:
        entry = material_data.get(str(mid), {})
        values.append(
            float(
                entry.get(
                    "scaling_factor",
                    entry.get("total_shift", 1.0),
                )
            )
        )
    return np.asarray(values, dtype=np.float32)


def _coords_for_nodes(node_coords, node_ids):
    coords = []
    for node in node_ids:
        key = str(node)
        if key not in node_coords:
            raise KeyError(f"node_coords missing node {node}")
        coords.append(node_coords[key])
    return coords


def _stored_response_values(sample, response_keys):
    store = sample.get("array_store") or {}
    datasets = store.get("datasets", {})
    missing = [key for key in response_keys if key not in datasets]
    if missing:
        raise KeyError(f"sample missing stored response fields: {missing}")

    sample_path = sample.get("_sample_path")
    if not sample_path:
        raise KeyError("stored response is missing its source sample path")
    store_path = store.get("file")
    group_path = store.get("group")
    if not store_path or not group_path:
        raise KeyError("array_store requires file and group")
    if not os.path.isabs(store_path):
        store_path = os.path.join(os.path.dirname(sample_path), store_path)

    with h5py.File(store_path, "r") as h5_file:
        group = h5_file[group_path]
        values = {key: group[datasets[key]][:] for key in response_keys}

    verified = sample.setdefault("_verified_array_keys", set())
    integrity = sample.get("array_integrity", {})
    for key, array in values.items():
        expected = integrity.get(key)
        if not expected or key in verified:
            continue
        contiguous = np.ascontiguousarray(array)
        actual_hash = hashlib.sha256(contiguous.tobytes()).hexdigest()
        if list(contiguous.shape) != list(expected.get("shape", [])):
            raise ValueError(f"Stored {key} shape does not match array_integrity")
        if str(contiguous.dtype) != str(expected.get("dtype")):
            raise ValueError(f"Stored {key} dtype does not match array_integrity")
        if actual_hash != expected.get("sha256"):
            raise ValueError(f"Stored {key} failed SHA-256 integrity validation")
        verified.add(key)
    return values


def _response_value(sample, response_key):
    nested = _nested_response(sample, response_key)
    if nested is not None:
        return nested
    return _stored_response_values(sample, [response_key])[response_key]


def _has_response(sample, response_key):
    return (
        _nested_response(sample, response_key) is not None
        or response_key in sample.get("array_store", {}).get("datasets", {})
    )


def _as_response_array(sample, response_key, node_ids, *map_keys):
    arr = np.asarray(_response_value(sample, response_key), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"{response_key} must have shape (steps, nodes, channels), got {arr.shape}"
        )

    node_map = _response_map(sample, *(map_keys or (response_key,)))
    if node_map:
        try:
            indices = [int(node_map[str(node)]) for node in node_ids]
        except KeyError as exc:
            raise KeyError(
                f"{response_key} node map missing expected node {exc.args[0]}"
            ) from exc
        arr = arr[:, indices, :]
    elif arr.shape[1] != len(node_ids):
        raise ValueError(
            f"{response_key} has {arr.shape[1]} nodes, expected {len(node_ids)}"
        )
    return arr


def _temperature_condition(sample):
    temperature = sample.get("environment", {}).get("temperature", {}) or {}
    values = temperature.get("temperature_steps_C", [])
    value = temperature.get("value_C")
    if value is None:
        value = float(np.mean(values)) if values else temperature.get(
            "reference_temperature_C", 0.0
        )
    reference = float(temperature.get("reference_temperature_C", 0.0))
    delta = float(value) - reference
    return float(value), delta


class StructuralDataset(Dataset):
    """Load response tensors and condition/safety labels from result_*.json."""

    def __init__(self, data_dir, normalize=True, fit_normalizer=False, normalizer_indices=None):
        self.data_dir = data_dir
        files = sorted(glob.glob(os.path.join(data_dir, "result_*.json")))
        if not files:
            raise FileNotFoundError(f"No result_*.json files found in {data_dir}")

        manifest = {}
        manifest_path = os.path.join(data_dir, "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as file:
                manifest = json.load(file)

        samples = []
        for i, path in enumerate(files):
            with open(path, encoding="utf-8") as file:
                sample = json.load(file)
            sample["_sample_path"] = path
            for key in (
                "node_coords",
                "material_nodes",
                "node_maps",
                "region_definitions",
                "response_metadata",
            ):
                if key not in sample and key in manifest:
                    sample[key] = manifest[key]
            samples.append(sample)
            if (i + 1) % 50 == 0:
                print(f"  Loading metadata: {i + 1}/{len(files)}", flush=True)
        print(f"  Loaded metadata: {len(files)}/{len(files)}")

        meta = samples[0]

        self.material_ids = _material_ids_from_sample(meta)
        self.node_coords = meta["node_coords"]
        self.node_maps = meta.get("node_maps", {})
        self.static_node_ids = _nodes_from_sample(meta, "disp", STATIC_NODES, "disp")
        self.dynamic_node_ids = _nodes_from_sample(meta, "ace", DYNAMIC_NODES, "acc", "ace")

        support_meta = meta.get("support_settlement", {})
        self.support_nodes = [int(n) for n in support_meta.get("node_ids", SUPPORT_NODES)]
        safety_meta = meta.get("safety_labels", {})
        self.region_names = list(
            safety_meta.get(
                "region_names",
                meta.get("region_definitions", {}).keys() or REGION_NAMES,
            )
        )

        self.static_pos = torch.tensor(
            _coords_for_nodes(self.node_coords, self.static_node_ids), dtype=torch.float32
        )
        self.dynamic_pos = torch.tensor(
            _coords_for_nodes(self.node_coords, self.dynamic_node_ids), dtype=torch.float32
        )

        self.n_samples = len(samples)
        self.files = files
        self.samples = samples
        self.lazy_responses = all(
            sample.get("array_store") and "disp" not in sample and "ace" not in sample
            for sample in samples
        )

        first_disp, first_strain, first_ace = self.response_arrays(meta)
        self.static_steps, self.n_static_nodes, self.static_channels = first_disp.shape
        self.dynamic_steps, self.n_dynamic_nodes, self.dynamic_channels = first_ace.shape

        expected_disp_shape = list(first_disp.shape)
        expected_ace_shape = list(first_ace.shape)
        for sample in samples[1:]:
            response_metadata = sample.get("response_metadata", {})
            disp_shape = response_metadata.get("disp", {}).get("shape")
            ace_shape = response_metadata.get("ace", {}).get("shape")
            if disp_shape is not None and list(disp_shape) != expected_disp_shape:
                raise ValueError(
                    f"Inconsistent displacement shape: {disp_shape} != {expected_disp_shape}"
                )
            if ace_shape is not None and list(ace_shape) != expected_ace_shape:
                raise ValueError(
                    f"Inconsistent acceleration shape: {ace_shape} != {expected_ace_shape}"
                )

        if self.lazy_responses:
            self.disp = self.strain = self.ace = None
        else:
            disp, strain, ace = [first_disp], [first_strain], [first_ace]
            for sample in samples[1:]:
                sample_disp, sample_strain, sample_ace = self.response_arrays(sample)
                disp.append(sample_disp)
                strain.append(sample_strain)
                ace.append(sample_ace)
            self.disp = torch.tensor(np.asarray(disp), dtype=torch.float32)
            self.strain = torch.tensor(np.asarray(strain), dtype=torch.float32)
            self.ace = torch.tensor(np.asarray(ace), dtype=torch.float32)

        raw_sf = torch.tensor(
            np.array([
                _material_scaling_from_sample(s, self.material_ids) for s in samples
            ]),
            dtype=torch.float32,
        )
        cand_idx = [self.material_ids.index(m) for m in CANDIDATE_IDS]
        self.raw_sf = raw_sf[:, cand_idx]

        labels = [self._labels_for_sample(s) for s in samples]
        self.targets = torch.tensor(np.array([x["material_labels"] for x in labels]), dtype=torch.float32)
        self.support_disp = torch.tensor(np.array([x["support_disp"] for x in labels]), dtype=torch.float32)
        self.support_targets = torch.tensor(np.array([x["support_labels"] for x in labels]), dtype=torch.float32)
        self.region_targets = torch.tensor(np.array([x["region_risk_levels"] for x in labels]), dtype=torch.long)
        self.global_targets = torch.tensor(np.array([x["global_level"] for x in labels]), dtype=torch.long)
        self.material_risk_levels = torch.tensor(
            np.array([x["material_risk_levels"] for x in labels]), dtype=torch.long
        )
        self.support_risk_levels = torch.tensor(
            np.array([x["support_risk_levels"] for x in labels]), dtype=torch.long
        )
        conditions = [_temperature_condition(sample) for sample in samples]
        self.temperature_c = torch.tensor(
            [value for value, _delta in conditions], dtype=torch.float32
        )
        self.condition = torch.tensor(
            [[delta / 20.0] for _value, delta in conditions], dtype=torch.float32
        )
        self.temperature_steps_c = torch.tensor(
            [
                sample.get("environment", {})
                .get("temperature", {})
                .get("temperature_steps_C", [value])
                for sample, (value, _delta) in zip(samples, conditions)
            ],
            dtype=torch.float32,
        )

        self.group_damage_pattern = self._build_damage_pattern_groups()
        self.group_scaling_bin = self._build_scaling_bin_groups()
        self.group_file_block = self._build_file_block_groups()
        self.group_safety_state = self._build_safety_state_groups()
        self.group_structural_state = self._build_structural_state_groups()

        self.normalize = normalize
        self.disp_mean = self.strain_mean = self.ace_mean = torch.zeros(1)
        self.disp_std = self.strain_std = self.ace_std = torch.ones(1)
        if normalize and fit_normalizer:
            indices = range(self.n_samples) if normalizer_indices is None else normalizer_indices
            self.fit_normalizer(indices)

    def response_arrays(self, sample):
        local_sample = sample
        stored_keys = [
            key for key in ("disp", "strain", "ace")
            if key not in sample
            and _nested_response(sample, key) is None
            and _has_response(sample, key)
        ]
        if stored_keys:
            local_sample = dict(sample)
            local_sample.update(_stored_response_values(sample, stored_keys))

        disp = _as_response_array(local_sample, "disp", self.static_node_ids, "disp")
        if _has_response(local_sample, "strain"):
            strain = _as_response_array(
                local_sample, "strain", self.static_node_ids, "strain"
            )
        else:
            strain = np.zeros_like(disp, dtype=np.float32)
        ace = _as_response_array(
            local_sample, "ace", self.dynamic_node_ids, "acc", "ace"
        )
        return disp, strain, ace

    def make_input_tensors(self, sample, normalize=None):
        if normalize is None:
            normalize = self.normalize
        disp, strain, ace = self.response_arrays(sample)
        disp = torch.tensor(disp, dtype=torch.float32)
        strain = torch.tensor(strain, dtype=torch.float32)
        ace = torch.tensor(ace, dtype=torch.float32)
        if normalize:
            disp = self._normalize(disp.unsqueeze(0), self.disp_mean, self.disp_std).squeeze(0)
            strain = self._normalize(strain.unsqueeze(0), self.strain_mean, self.strain_std).squeeze(0)
            ace = self._normalize(ace.unsqueeze(0), self.ace_mean, self.ace_std).squeeze(0)
        return disp, strain, ace

    def apply_to_config(self, config):
        config.n_materials = len(CANDIDATE_IDS)
        config.n_supports = len(self.support_nodes)
        config.n_regions = len(self.region_names)
        config.static_steps = self.static_steps
        config.static_channels = self.static_channels
        config.n_static_nodes = self.n_static_nodes
        config.dynamic_channels = self.dynamic_channels
        config.n_dynamic_nodes = self.n_dynamic_nodes
        config.static_node_ids = list(self.static_node_ids)
        config.dynamic_node_ids = list(self.dynamic_node_ids)
        config.support_nodes = list(self.support_nodes)
        config.region_names = list(self.region_names)

    def _labels_for_sample(self, sample):
        support = sample.get("support_settlement", {})
        support_values = support.get("values_mm", [0.0] * len(self.support_nodes))
        safety = sample.get("safety_labels")
        if safety is None:
            material_ids = _material_ids_from_sample(sample)
            kwargs = {}
            if sample.get("region_definitions"):
                kwargs["region_definitions"] = sample["region_definitions"]
            safety = build_safety_labels(
                material_ids=material_ids,
                material_scaling_factors=_material_scaling_from_sample(sample, material_ids),
                support_settlement_mm=support_values,
                **kwargs,
            )
        material_ids = _material_ids_from_sample(sample)
        material_sf = _material_scaling_from_sample(sample, material_ids)
        return {
            "material_labels": safety.get(
                "material_labels",
                (material_sf[
                    [material_ids.index(m) for m in CANDIDATE_IDS]
                ] < 1.0).astype(float).tolist(),
            ),
            "material_risk_levels": safety.get("material_risk_levels", [0] * len(CANDIDATE_IDS)),
            "support_disp": support_values,
            "support_labels": safety.get("support_labels", [0] * len(self.support_nodes)),
            "support_risk_levels": safety.get("support_risk_levels", [0] * len(self.support_nodes)),
            "region_risk_levels": safety.get("region_risk_levels", [0] * len(self.region_names)),
            "global_level": safety.get("global_level", 0),
        }

    @staticmethod
    def _compute_stats(tensor, indices=None, dynamic=False):
        source = tensor if indices is None else tensor[list(indices)]
        if dynamic:
            mean = source.mean(dim=(0, 1), keepdim=True)
            std = source.std(dim=(0, 1), keepdim=True, unbiased=False)
        else:
            mean = source.mean(dim=0, keepdim=True)
            std = source.std(dim=0, keepdim=True, unbiased=False)
        return mean, std.clamp(min=1e-8)

    def _fit_lazy_normalizer(self, indices):
        indices = [int(index) for index in indices]
        if not indices:
            raise ValueError("normalizer indices cannot be empty")

        disp_sum = disp_sq = strain_sum = strain_sq = None
        ace_sum = ace_sq = None
        static_count = 0
        dynamic_count = 0
        for index in indices:
            disp, strain, ace = self.response_arrays(self.samples[index])
            if disp_sum is None:
                disp_sum = np.zeros_like(disp, dtype=np.float64)
                disp_sq = np.zeros_like(disp, dtype=np.float64)
                strain_sum = np.zeros_like(strain, dtype=np.float64)
                strain_sq = np.zeros_like(strain, dtype=np.float64)
                ace_sum = np.zeros(ace.shape[1:], dtype=np.float64)
                ace_sq = np.zeros(ace.shape[1:], dtype=np.float64)
            disp_sum += disp
            disp_sq += np.square(disp, dtype=np.float64)
            strain_sum += strain
            strain_sq += np.square(strain, dtype=np.float64)
            ace_sum += ace.sum(axis=0, dtype=np.float64)
            ace_sq += np.square(ace, dtype=np.float64).sum(axis=0)
            static_count += 1
            dynamic_count += ace.shape[0]

        def moments(total, total_sq, count, prefix_shape):
            mean = total / count
            variance = np.maximum(total_sq / count - mean ** 2, 1e-16)
            return (
                torch.tensor(mean.reshape(prefix_shape + mean.shape), dtype=torch.float32),
                torch.tensor(np.sqrt(variance).reshape(prefix_shape + mean.shape), dtype=torch.float32),
            )

        self.disp_mean, self.disp_std = moments(disp_sum, disp_sq, static_count, (1,))
        self.strain_mean, self.strain_std = moments(
            strain_sum, strain_sq, static_count, (1,)
        )
        self.ace_mean, self.ace_std = moments(ace_sum, ace_sq, dynamic_count, (1, 1))

    def fit_normalizer(self, indices):
        if not self.normalize:
            return
        if self.lazy_responses:
            self._fit_lazy_normalizer(indices)
            return
        self.disp_mean, self.disp_std = self._compute_stats(self.disp, indices)
        self.strain_mean, self.strain_std = self._compute_stats(self.strain, indices)
        self.ace_mean, self.ace_std = self._compute_stats(
            self.ace, indices, dynamic=True
        )

    def normalizer_state_dict(self):
        return {
            "disp_mean": self.disp_mean.cpu(),
            "disp_std": self.disp_std.cpu(),
            "strain_mean": self.strain_mean.cpu(),
            "strain_std": self.strain_std.cpu(),
            "ace_mean": self.ace_mean.cpu(),
            "ace_std": self.ace_std.cpu(),
        }

    def load_normalizer_state_dict(self, state):
        if not state:
            return
        self.disp_mean = state["disp_mean"].cpu()
        self.disp_std = state["disp_std"].cpu()
        self.strain_mean = state["strain_mean"].cpu()
        self.strain_std = state["strain_std"].cpu()
        self.ace_mean = state["ace_mean"].cpu()
        self.ace_std = state["ace_std"].cpu()

    def _normalize(self, tensor, mean, std):
        if mean.numel() == 1:
            return tensor
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
        if mean.dim() == 1 and mean.numel() == tensor[0].numel():
            orig_shape = tensor.shape
            flat = tensor.reshape(orig_shape[0], -1)
            return ((flat - mean) / std).reshape(orig_shape)
        return (tensor - mean) / std

    def _build_damage_pattern_groups(self):
        return ["".join(str(int(v)) for v in row.tolist()) for row in self.targets]

    def _build_scaling_bin_groups(self, n_bins=4):
        clipped = self.raw_sf.clamp(0.0, 1.0)
        bins = torch.clamp((clipped * n_bins).long(), max=n_bins - 1)
        return ["_".join(str(int(v)) for v in row.tolist()) for row in bins]

    def _build_file_block_groups(self, block_size=50):
        return [str(i // block_size) for i in range(self.n_samples)]

    def _build_safety_state_groups(self):
        return [str(int(v)) for v in self.global_targets.tolist()]

    def _build_structural_state_groups(self):
        groups = []
        for index, sample in enumerate(self.samples):
            sample_index = sample.get("sample_index", {})
            groups.append(
                str(
                    sample_index.get(
                        "structural_state_id",
                        sample_index.get("group_id", index),
                    )
                )
            )
        return groups

    def get_group_labels(self, group_by="damage_pattern"):
        groups = {
            "damage_pattern": self.group_damage_pattern,
            "scaling_bin": self.group_scaling_bin,
            "file_block": self.group_file_block,
            "safety_state": self.group_safety_state,
            "structural_state": self.group_structural_state,
        }
        if group_by not in groups:
            raise ValueError(f"unknown group_by: {group_by}")
        return groups[group_by]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.lazy_responses:
            disp_array, strain_array, ace_array = self.response_arrays(self.samples[idx])
            disp = torch.tensor(disp_array, dtype=torch.float32)
            strain = torch.tensor(strain_array, dtype=torch.float32)
            ace = torch.tensor(ace_array, dtype=torch.float32)
        else:
            disp = self.disp[idx]
            strain = self.strain[idx]
            ace = self.ace[idx]

        if self.normalize:
            disp = self._normalize(disp.unsqueeze(0), self.disp_mean, self.disp_std).squeeze(0)
            strain = self._normalize(strain.unsqueeze(0), self.strain_mean, self.strain_std).squeeze(0)
            ace = self._normalize(ace.unsqueeze(0), self.ace_mean, self.ace_std).squeeze(0)

        return {
            "disp": disp,
            "strain": strain,
            "ace": ace,
            "target": self.targets[idx],
            "raw_sf": self.raw_sf[idx],
            "support_disp": self.support_disp[idx],
            "support_target": self.support_targets[idx],
            "region_target": self.region_targets[idx],
            "global_target": self.global_targets[idx],
            "condition": self.condition[idx],
            "temperature_C": self.temperature_c[idx],
            "temperature_steps_C": self.temperature_steps_c[idx],
        }
