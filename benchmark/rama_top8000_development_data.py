#!/usr/bin/env python3
"""Prepare the six Top8000 two-angle development populations, without fitting.

The sampling unit in the saved arrays is an eligible residue, not one residue
per structure. Every PDB code receives one role before any angles are inspected.
Caps retain the lowest identifier hashes within that role/category. This keeps
the preparation bounded without preferentially sampling dense angular regions.
PDB separation does not assert independence of homologous molecular structures.
Recorded categories are preserved. In particular, the supplied IleVal labels
include some pre-Pro residues: they do not identify the published no-PreP class.

Run from any directory (all default paths are relative to this source file):
    python /path/to/ramachandran_plots/benchmark/rama_top8000_development_data.py

Only Top8000 is an input to this preparation. Existing output is immutable: a
repeat invocation verifies its input, source, settings and artifact hashes.
Equal two-angle duplicates coalesce. Every row at a position with conflicting
two-angle payloads is excluded and preserved in conflicting_positions.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import tempfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
CATEGORIES = ("General", "Gly", "IleVal", "PrePro", "TransPro", "CisPro")
GROUP_ROLES = ("fit", "selection_calibration", "final_calibration", "assessment")
SAVED_ROLES = ("fit", "refit", "selection_calibration", "final_calibration", "assessment")
VERSION = "top8000_pdb_separated_residue_preparation_v2"
SEED = 2026092361
# Basenames only. The first four characters must be a legacy PDB accession;
# a suffix must start at a delimiter, not extend that accession ambiguously.
_FILENAME = re.compile(r"[1-9][a-z0-9]{3}(?:[._-][a-z0-9_.-]+)?\.pdb(?:\.gz)?\Z")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources():
    return {"benchmark/" + Path(__file__).name: _sha256(Path(__file__))}


def _hash(seed, purpose, value):
    return hashlib.sha256(_json([VERSION, seed, purpose, value]).encode("utf-8")).digest()


def _role_from_digest(digest):
    # Integer comparison makes the prescribed rational cutoffs unambiguous.
    if not isinstance(digest, bytes) or len(digest) != 32:
        raise ValueError("Expected a 256-bit SHA256 digest")
    rank = int.from_bytes(digest, "big") * 20
    scale = 1 << 256
    for cutoff, role in zip((10, 14, 17, 20), GROUP_ROLES):
        if rank < cutoff * scale:
            return role
    raise ValueError("Expected a 256-bit SHA256 digest")


def structure_role(pdb, *, seed=SEED):
    """Return a canonical PDB group's role, independently of residue angles.

    Hash the UTF-8 canonical JSON array
    ``[VERSION, seed, 'pdb-role', pdb]`` with SHA256; compare its unsigned
    big-endian integer to the exact 50%, 70% and 85% cutoffs. PDB codes must
    already be canonical lowercase four-character legacy accessions.
    """
    if not isinstance(pdb, str) or not re.fullmatch(r"[1-9][a-z0-9]{3}", pdb):
        raise ValueError("pdb must be a canonical lowercase legacy PDB accession")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return _role_from_digest(_hash(seed, "pdb-role", pdb))


def _canonical_filename(value):
    if not isinstance(value, str) or not _FILENAME.fullmatch(value.lower()):
        raise ValueError("file must be a basename with a four-character legacy PDB accession and .pdb[.gz] suffix")
    return value.lower()


def _identity(record):
    filename = _canonical_filename(record.get("file"))
    for key in ("model", "resnum"):
        if type(record.get(key)) is not int:
            raise ValueError(f"{key} must be an integer (not Boolean)")
    for key in ("chain", "ins"):
        if not isinstance(record.get(key), str):
            raise ValueError(f"{key} must be a string")
    if len(record["ins"]) > 1:
        raise ValueError("ins must contain the actual zero- or one-character insertion code")
    if not isinstance(record.get("resname"), str) or not record["resname"]:
        raise ValueError("resname must be a nonempty string")
    return filename, _json([filename, record["model"], record["chain"], record["resnum"], record["ins"]])


def _angle(record, key):
    """Return a duplicate-comparison token and, if finite, a Float64 angle.

    Missing and null remain distinct. Invalid values remain in the duplicate
    payload, so changing eligibility at an existing position cannot be hidden
    by filtering. Numeric 1 and 1.0 represent the same measured Float64 angle.
    Chi angles and other fields never enter this explicitly two-angle payload.
    """
    if key not in record:
        return ["missing_key"], None
    value = record[key]
    if value is None:
        return ["null"], None
    if type(value) not in (int, float):
        return ["bad_type", json.dumps(value, sort_keys=True, ensure_ascii=False)], None
    try:
        angle = float(value)
    except OverflowError:
        return ["outside_float64_range", str(value)], None
    if not math.isfinite(angle):
        return ["nonfinite_number", str(value)], None
    return ["finite", angle if angle else 0.0], angle


def _spec(seed, fit_cap, refit_cap, role_cap):
    for name, value in (("seed", seed), ("fit_cap", fit_cap), ("refit_cap", refit_cap), ("role_cap", role_cap)):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
    return dict(version=VERSION, categories=list(CATEGORIES), dimension=2, angles=["phi", "psi"],
        periods=[360., 360.], origins=[-180., -180.], seed=seed,
        caps=dict(fit=fit_cap, refit=refit_cap, selection_calibration=role_cap,
                  final_calibration=role_cap, assessment=role_cap),
        group="first_four_characters_of_canonical_source_filename",
        filename_normalization="lowercase only; strict basename; legacy PDB [1-9][a-z0-9]{3}; .pdb or .pdb.gz",
        identity_fields=["canonical_file", "model", "chain", "resnum", "ins"],
        duplicate_payload=["resname", "rama_category", "phi_state_and_value", "psi_state_and_value"],
        duplicate_policy="coalesce equal 2D payloads; exclude EVERY row at an ambiguous positional identity before eligibility filtering; ignore chi-only differences",
        ambiguous_positions="preserve every source row, source line and distinct 2D payload in conflicting_positions.jsonl; never choose first or last variant",
        eligibility="one of the six categories and finite numeric non-Boolean phi and psi; no chi1 requirement",
        category_semantics="recorded rama_category; IleVal includes some pre-Pro residues and is not the published IleVal(noPreP) population",
        category_relabeling="none; no peptide connectivity inferred from residue order or numbering",
        category_comparison_scope="development on recorded classes; published-class transfer settings cannot be frozen from this preparation alone",
        role_probabilities=dict(zip(GROUP_ROLES, [.50, .20, .15, .15])),
        role_rule="SHA256 canonical JSON [version,seed,'pdb-role',pdb]; integer quantile cutoffs 10/20,14/20,17/20",
        sampling_rule="lowest SHA256 canonical JSON [version,seed,'residue-sample',canonical_identity_string]; tie break by identity",
        refit_pool="all eligible fit plus selection_calibration residues before either role's cap; apply refit cap afresh",
        refit_nesting="not guaranteed to retain every initially selected fit residue",
        population="eligible residue observations; coordinate-independent hash subsampling within each category and role",
        coverage_scope="PDB-separated molecular sampling; no IID or homology-independence guarantee",
        role_allocation="hash probabilities, not exact observed role proportions")


def _stat_token(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _checked_input_hash(path):
    before = _stat_token(path)
    digest = _sha256(path)
    if _stat_token(path) != before:
        raise RuntimeError("Input changed while its hash was computed")
    return digest, before[0]


def _resume(data_path, out, spec, sources):
    manifest_path = out / "preparation.json"
    if not manifest_path.is_file():
        raise ValueError("Output exists without preparation.json; refusing to overwrite partial or unfamiliar output")
    manifest = json.loads(manifest_path.read_text())
    if manifest["specification"] != spec or manifest["source_sha256"] != sources:
        raise ValueError("Existing preparation has different settings or source hashes")
    digest, size = _checked_input_hash(data_path)
    if digest != manifest["input"]["sha256"] or size != manifest["input"]["bytes"]:
        raise ValueError("Input differs from the immutable saved preparation")
    expected_datasets = {f"categories/{category}/observations.npz" for category in CATEGORIES}
    if [case["category"] for case in manifest["cases"]] != list(CATEGORIES):
        raise ValueError("Saved category manifest is malformed")
    if {case["dataset"] for case in manifest["cases"]} != expected_datasets:
        raise ValueError("Saved category dataset paths are malformed")
    for relative, expected in manifest["artifact_sha256"].items():
        path = (out / relative).resolve()
        if not path.is_relative_to(out.resolve()) or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Saved artifact failed verification: {relative}")
    for case in manifest["cases"]:
        if manifest["artifact_sha256"].get(case["dataset"]) != case["dataset_sha256"]:
            raise ValueError("Dataset hashes disagree inside preparation.json")
    required = expected_datasets | {"duplicate_ids.jsonl", "conflicting_positions.jsonl", "pdb_groups.json"}
    if set(manifest["artifact_sha256"]) != required:
        raise ValueError("Saved artifact manifest is incomplete or unfamiliar")
    if _sources() != sources:
        raise RuntimeError("Preparation source changed during verification")
    return manifest


def _new_database(path):
    db = sqlite3.connect(path)
    # This is disposable deduplication state, not a persisted scientific result.
    # Keep its page cache bounded and its large sort/index work on disk.
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("""CREATE TABLE residues (
        identity TEXT PRIMARY KEY, payload TEXT NOT NULL, file TEXT NOT NULL,
        pdb TEXT NOT NULL, role TEXT NOT NULL, category TEXT,
        phi REAL, psi REAL, phi_state TEXT NOT NULL, psi_state TEXT NOT NULL,
        eligible INTEGER NOT NULL, rank BLOB NOT NULL, copies INTEGER NOT NULL DEFAULT 1,
        ambiguous INTEGER NOT NULL DEFAULT 0
    )""")
    return db


def _scan(data_path, db, seed):
    before = _stat_token(data_path)
    digest = hashlib.sha256()
    counters = Counter(lines=0, blank_lines=0, input_rows=0, filename_case_normalized_rows=0,
                       eligible_input_rows_before_conflict_exclusion=0)
    raw_category_counts = Counter({category: 0 for category in CATEGORIES})
    role_cache = {}  # A few thousand PDB groups, never a dictionary of residues.
    with data_path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            counters["lines"] += 1
            if not line.strip():
                counters["blank_lines"] += 1
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON row must be an object")
                filename, identity = _identity(record)
                phi_token, phi = _angle(record, "phi")
                psi_token, psi = _angle(record, "psi")
                category = record.get("rama_category")
                category_token = ["missing_key"] if "rama_category" not in record else ["value", category]
                payload = _json([record["resname"], category_token, phi_token, psi_token])
                eligible = isinstance(category, str) and category in CATEGORIES and phi is not None and psi is not None
                pdb = filename[:4]
                if pdb not in role_cache:
                    role_cache[pdb] = structure_role(pdb, seed=seed)
                role = role_cache[pdb]
                category_sql = category if isinstance(category, str) else None
                inserted = db.execute("INSERT OR IGNORE INTO residues VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,0)",
                    (identity, payload, filename, pdb, role, category_sql, phi, psi,
                     phi_token[0], psi_token[0], int(eligible), _hash(seed, "residue-sample", identity)))
                if inserted.rowcount == 0:
                    old_payload = db.execute("SELECT payload FROM residues WHERE identity=?", (identity,)).fetchone()[0]
                    if old_payload != payload:
                        # This position cannot contribute any variant. Keep its
                        # first payload only for comparison, never as a selected
                        # observation; the second input pass preserves all rows.
                        db.execute("UPDATE residues SET ambiguous=1,eligible=0 WHERE identity=?", (identity,))
                    db.execute("UPDATE residues SET copies=copies+1 WHERE identity=?", (identity,))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                raise ValueError(f"Input line {line_number}: {error}") from error
            counters["input_rows"] += 1
            counters["filename_case_normalized_rows"] += filename != record["file"]
            counters["eligible_input_rows_before_conflict_exclusion"] += bool(eligible)
            if eligible:
                raw_category_counts[category] += 1
            if counters["input_rows"] % 10000 == 0:
                db.commit()
            if counters["input_rows"] % 250000 == 0:
                print(f"Prepared identity index for {counters['input_rows']:,} input rows", file=sys.stderr, flush=True)
    db.commit()
    if _stat_token(data_path) != before:
        raise RuntimeError("Input changed during preparation")
    if counters["input_rows"] == 0:
        raise ValueError("Input contains no observation rows")
    db.execute("CREATE INDEX population_sampling ON residues(category,role,rank,identity) WHERE eligible=1")
    db.commit()
    return dict(counters, raw_category_eligible_input_rows=dict(raw_category_counts)), digest.hexdigest(), before[0]


def _counts(db, where="eligible=1", parameters=()):
    rows = db.execute(f"""SELECT COUNT(*),COUNT(DISTINCT pdb),COUNT(DISTINCT file),
        COALESCE(SUM(copies),0),COALESCE(SUM(CASE WHEN ambiguous=0 THEN copies-1 ELSE 0 END),0)
        FROM residues WHERE {where}""", parameters).fetchone()
    return dict(observations=rows[0], pdb_groups=rows[1], source_files=rows[2],
                input_rows=rows[3], duplicate_rows_coalesced=rows[4])


def _save_category(db, stage, category, caps):
    arrays, roles = {}, {}
    for role in SAVED_ROLES:
        pool = ("fit", "selection_calibration") if role == "refit" else (role,)
        placeholders = ",".join("?" for _ in pool)
        where = f"eligible=1 AND category=? AND role IN ({placeholders})"
        parameters = (category, *pool)
        counts = _counts(db, where, parameters)
        selected = db.execute(f"""SELECT phi,psi,pdb,identity,file FROM residues WHERE {where}
            ORDER BY rank,identity LIMIT ?""", (*parameters, caps[role])).fetchall()
        points = np.array([[row[0], row[1]] for row in selected], dtype=np.float64).reshape(-1, 2)
        # Wrap only after duplicate comparison, so discrepant raw measurements
        # cannot silently become equal by a periodic-coordinate transformation.
        arrays[role + "_points"] = (points + 180.) % 360. - 180.
        for column, suffix in ((2, "groups"), (3, "ids"), (4, "source_files")):
            arrays[role + "_" + suffix] = np.array([row[column] for row in selected], dtype=str)
        roles[role] = dict(pool_observations=counts["observations"], selected_observations=len(selected),
            pool_pdb_groups=counts["pdb_groups"], selected_pdb_groups=len(set(arrays[role + "_groups"].tolist())),
            pool_source_files=counts["source_files"], selected_source_files=len(set(arrays[role + "_source_files"].tolist())))
    relative = f"categories/{category}/observations.npz"
    destination = stage / relative
    destination.parent.mkdir(parents=True)
    np.savez_compressed(destination, **arrays)
    return dict(category=category, dataset=relative, dataset_sha256=_sha256(destination),
        population_counts=_counts(db, "eligible=1 AND category=?", (category,)), roles=roles)


def _save_indexes(db, stage):
    duplicate_positions = 0
    with (stage / "duplicate_ids.jsonl").open("w", encoding="utf-8") as stream:
        for identity, copies, category, eligible in db.execute(
                "SELECT identity,copies,category,eligible FROM residues WHERE copies>1 AND ambiguous=0 ORDER BY identity"):
            stream.write(_json(dict(identity=identity, input_copies=copies, duplicate_rows_coalesced=copies-1,
                                    category=category, eligible=bool(eligible))) + "\n")
            duplicate_positions += 1
    groups = [dict(pdb_group=pdb, role=role, all_unique_positions=all_count,
                   eligible_observations=eligible, all_source_files=files, eligible_source_files=eligible_files)
        for pdb, role, all_count, eligible, files, eligible_files in db.execute("""
            SELECT pdb,role,COUNT(*),SUM(eligible),COUNT(DISTINCT file),
                   COUNT(DISTINCT CASE WHEN eligible=1 THEN file END)
            FROM residues GROUP BY pdb,role ORDER BY pdb""")]
    (stage / "pdb_groups.json").write_text(_json(groups) + "\n", encoding="utf-8")
    return duplicate_positions


def _save_conflicts(data_path, db, stage, input_hash):
    """Recover every original row at excluded positions, using bounded memory.

    A second streaming pass avoids retaining a million full input dictionaries.
    Only ambiguous positions' rows enter the small temporary SQLite ledger.
    Rehashing also verifies that these rows come from the same input snapshot.
    """
    conflict_files = {row[0] for row in db.execute("SELECT DISTINCT file FROM residues WHERE ambiguous=1")}
    db.execute("CREATE TEMP TABLE conflict_lines (identity TEXT,line_number INTEGER,payload TEXT,source_json TEXT)")
    digest = hashlib.sha256()
    if conflict_files:
        with data_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["file"].lower() not in conflict_files:
                    continue
                _, identity = _identity(record)
                ambiguous = db.execute("SELECT ambiguous FROM residues WHERE identity=?", (identity,)).fetchone()[0]
                if not ambiguous:
                    continue
                phi, _ = _angle(record, "phi"); psi, _ = _angle(record, "psi")
                category = ["value", record["rama_category"]] if "rama_category" in record else ["missing_key"]
                payload = _json([record["resname"], category, phi, psi])
                db.execute("INSERT INTO conflict_lines VALUES (?,?,?,?)",
                    (identity, line_number, payload, line.decode().rstrip("\r\n")))
        if digest.hexdigest() != input_hash:
            raise RuntimeError("Input changed while preserving ambiguous positions")
    db.execute("CREATE INDEX conflict_identity ON conflict_lines(identity)")
    positions = input_rows = eligible_input_rows = variants = 0
    excluded_by_category = Counter({category: 0 for category in CATEGORIES})
    with (stage / "conflicting_positions.jsonl").open("w", encoding="utf-8") as stream:
        for (identity,) in db.execute("SELECT identity FROM residues WHERE ambiguous=1 ORDER BY identity"):
            rows = db.execute("SELECT line_number,payload,source_json FROM conflict_lines WHERE identity=? ORDER BY line_number", (identity,)).fetchall()
            payloads = [json.loads(value) for value in sorted({row[1] for row in rows})]
            if len(payloads) < 2:
                raise RuntimeError("Excluded positional identity lacks its conflicting source payloads")
            for _, payload, _ in rows:
                value = json.loads(payload)
                if value[1][0] == "value" and value[1][1] in CATEGORIES and value[2][0] == value[3][0] == "finite":
                    eligible_input_rows += 1
                    excluded_by_category[value[1][1]] += 1
            stream.write(_json(dict(identity=identity, payload_variants=payloads,
                source_rows=[dict(line_number=number, source_json=raw) for number, _, raw in rows])) + "\n")
            positions += 1; input_rows += len(rows); variants += len(payloads)
    return dict(positions=positions, input_rows=input_rows, payload_variants=variants,
                eligible_input_rows_excluded=eligible_input_rows,
                eligible_input_rows_excluded_by_category=dict(excluded_by_category))


def prepare_top8000(data_path, out, *, seed=SEED, fit_cap=20000, refit_cap=20000, role_cap=40000):
    """Save bounded 2D residue samples and a complete population-count manifest.

    Initial fit and selection calibration belong to separate PDB groups. Refit
    selects afresh from their full union, then applies its own cap. Thus a later
    fit may replace some initially selected sites. Final calibration and
    assessment remain separate from both fitting pools. All six categories use
    the same PDB-role rule. No contour method is called here.
    """
    data_path, out = Path(data_path).resolve(), Path(out).resolve()
    spec = _spec(seed, fit_cap, refit_cap, role_cap)
    sources = _sources()
    if out.exists():
        return _resume(data_path, out, spec, sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    input_stat = _stat_token(data_path)
    # Publish only a complete preparation. The disposable SQLite index can be
    # large, but never becomes part of the portable saved research artifacts.
    with tempfile.TemporaryDirectory(prefix=".top8000-preparation-", dir=out.parent) as temporary:
        temporary = Path(temporary)
        stage = temporary / "saved"
        stage.mkdir()
        db = _new_database(temporary / "identities.sqlite")
        try:
            input_counts, input_hash, input_bytes = _scan(data_path, db, seed)
            conflicts = _save_conflicts(data_path, db, stage, input_hash)
            cases = [_save_category(db, stage, category, spec["caps"]) for category in CATEGORIES]
            duplicate_positions = _save_indexes(db, stage)
            population = _counts(db)
            all_positions = _counts(db, "1=1")
            global_roles = {role: _counts(db, "eligible=1 AND role=?", (role,)) for role in GROUP_ROLES}
            exclusions = [dict(category=category, phi_state=phi, psi_state=psi, observations=count,
                               input_rows=raw_count)
                for category, phi, psi, count, raw_count in db.execute("""
                    SELECT category,phi_state,psi_state,COUNT(*),SUM(copies) FROM residues
                    WHERE eligible=0 AND ambiguous=0 GROUP BY category,phi_state,psi_state
                    ORDER BY category,phi_state,psi_state""")]
        finally:
            db.close()
        if _sources() != sources:
            raise RuntimeError("Preparation source changed during execution")
        if _stat_token(data_path) != input_stat:
            raise RuntimeError("Input changed before publication of the preparation")
        artifacts = {str(path.relative_to(stage)): _sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        try:
            input_name = str(data_path.relative_to(PROJECT))
        except ValueError:
            input_name = str(data_path)
        manifest = dict(status="prepared", specification=spec, source_sha256=sources,
            input=dict(path=input_name, sha256=input_hash, bytes=input_bytes, **input_counts),
            environment=dict(python=platform.python_version(), numpy=np.__version__, sqlite=sqlite3.sqlite_version),
            population_counts=population, all_position_counts=all_positions,
            duplicate_positions=duplicate_positions, ambiguous_position_exclusions=conflicts,
            exclusions=exclusions, roles=global_roles,
            cases=cases, artifact_sha256=artifacts)
        (stage / "preparation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        # Recheck the output to refuse races with another preparer.
        if out.exists():
            raise ValueError("Output appeared while preparing; refusing to replace it")
        os.rename(stage, out)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data_points/top8000_measures.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT / "novak_work/validation_results/top8000_development_data_v2")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fit-cap", type=int, default=20000)
    parser.add_argument("--refit-cap", type=int, default=20000)
    parser.add_argument("--role-cap", type=int, default=40000)
    args = parser.parse_args()
    manifest = prepare_top8000(args.data, args.out, seed=args.seed, fit_cap=args.fit_cap,
                               refit_cap=args.refit_cap, role_cap=args.role_cap)
    print(json.dumps(dict(status=manifest["status"], manifest=str(args.out / "preparation.json"),
        population_counts=manifest["population_counts"], cases=[dict(category=case["category"], roles=case["roles"])
            for case in manifest["cases"]]), indent=2))


if __name__ == "__main__":
    main()
