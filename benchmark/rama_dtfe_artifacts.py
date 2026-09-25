"""Strict portable storage for saved DTFE families and development observations.

This owner handles artifact identity and query I/O only. Julia remains the
owner of the certified triangulation, density estimator and closed-triangle
score. Statistical calibration is unchanged in rama_contour_protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from rama_top8000_development_data import structure_role

PROJECT = Path(__file__).resolve().parents[1]
CAPSULE = PROJECT / "novak_work/validation_results/dtfe_development_v1"
CATEGORIES = ("CisPro", "TransPro", "Gly", "General")
SETTINGS = dict(sigma_deg=[0., 4., 8., 12., 16.], area_floor_quantiles=[0., .01],
                radius_rules=["native_shared", "no_cutoff"])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN is not a scientific result")
        return "Inf" if value > 0 else "-Inf"
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def local_path(name, base=PROJECT):
    name, base = Path(name), Path(base).resolve()
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("A canonical relative artifact path is required")
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Artifact escapes its declared directory")
    return path


def check_sources(sources, *, base=PROJECT):
    for relative, expected in sources.items():
        if sha256_file(local_path(relative, base)) != expected:
            raise ValueError("Bound source/artifact changed: " + relative)


def artifact_inventory(directory, *, exclude=("terminal.json",)):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and str(p.relative_to(directory)) not in exclude}


def candidate_ids():
    return [f"sigma{int(s)}_floor{'0' if f == 0 else '0p01'}_{r}"
            for s in SETTINGS["sigma_deg"] for f in SETTINGS["area_floor_quantiles"]
            for r in SETTINGS["radius_rules"]]


def load_development_capsule(directory=CAPSULE):
    """Verify the self-contained DTFE capsule; historical originals are not read."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["schema"] != "dtfe_development_capsule_v1"
            or manifest["categories"] != list(CATEGORIES)
            or manifest["candidate_ids"] != candidate_ids()
            or set(manifest["cases"]) != set(CATEGORIES)
            or manifest["artifacts"] != artifact_inventory(directory, exclude=("manifest.json",))):
        raise ValueError("DTFE capsule schema, candidate order, or artifact inventory differs")
    check_sources(manifest["sources"])
    for category, case in manifest["cases"].items():
        roles_path = local_path(case["dataset"], directory)
        if sha256_file(roles_path) != case["dataset_sha256"]:
            raise ValueError("Development observation artifact changed")
        family = json.loads((directory / "families" / category / "family.json").read_text())
        if [row["id"] for row in family["candidates"]] != candidate_ids():
            raise ValueError("Saved Julia candidate order differs")
    return manifest


def load_roles(case, *, base=CAPSULE):
    """Load the original fit/calibration/selection rows, preserving PDB roles."""
    path = local_path(case["dataset"], base)
    if sha256_file(path) != case["dataset_sha256"]:
        raise ValueError("Prepared observation hash differs")
    roles, group_sets = {}, {}
    with np.load(path, allow_pickle=False) as data:
        for role in ("refit", "final_calibration", "assessment"):
            points, groups, ids = (data[role + suffix] for suffix in ("_points", "_groups", "_ids"))
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                    or groups.shape != (len(points),) or ids.shape != groups.shape
                    or groups.dtype.kind != "U" or ids.dtype.kind != "U" or len(set(ids)) != len(ids)):
                raise ValueError("Malformed prepared DTFE development role")
            permitted = ("fit", "selection_calibration") if role == "refit" else (role,)
            if any(structure_role(g, seed=2026092361) not in permitted for g in set(groups)):
                raise ValueError("PDB assignment differs from the declared preparation")
            roles[role] = dict(points=points, groups=groups, ids=ids)
            group_sets[role] = set(groups)
    if any(group_sets[a] & group_sets[b] for a in group_sets for b in group_sets if a != b):
        raise ValueError("Development fit, calibration and selection PDB groups overlap")
    return roles


def write_query_points(directory, points, *, identities=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=">f8")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Finite phi/psi pairs are required")
    path = directory / "points.f64"
    points.tofile(path)
    metadata = dict(path=path.name, shape=list(points.shape), dtype=">f8", order="C",
                    axes=["observation", "angle"], units="degrees_phi_psi", bytes=points.nbytes, sha256=sha256_file(path))
    result = dict(file=str(path.relative_to(directory.parent)), metadata=metadata)
    if identities is not None:
        if len(identities["ids"]) != len(points) or len(identities["groups"]) != len(points):
            raise ValueError("One original identity/group per query row is required")
        identity_path = directory / "identities.json"
        write_json(identity_path, dict(groups=identities["groups"].tolist(), ids=identities["ids"].tolist(),
            identity_observations=len(identities["ids"]), coordinate_rows=len(points),
            scope="Original residue queries; no symmetry augmentation of calibration or selection observations"))
        result.update(identities_file=str(identity_path.relative_to(directory.parent)), identities_sha256=sha256_file(identity_path))
    return result


def read_f64(directory, metadata, *, shape, axes, units, allow_inf=False):
    path = local_path(metadata["path"], directory)
    if (set(metadata) != {"path", "shape", "dtype", "order", "axes", "units", "bytes", "sha256"}
            or Path(metadata["path"]).name != metadata["path"]
            or metadata["shape"] != list(shape) or metadata["axes"] != axes
            or metadata["units"] != units or metadata["dtype"] != ">f8" or metadata["order"] != "C"
            or path.stat().st_size != 8 * int(np.prod(shape)) or metadata["bytes"] != path.stat().st_size
            or sha256_file(path) != metadata["sha256"]):
        raise ValueError("Portable incidence artifact metadata, bytes, or fingerprint differ")
    values = np.fromfile(path, dtype=">f8").reshape(shape).astype(float)
    if np.isnan(values).any() or np.isneginf(values).any() or (not allow_inf and not np.isfinite(values).all()):
        raise ValueError("Nonfinite portable values violate the declared contract")
    return values

