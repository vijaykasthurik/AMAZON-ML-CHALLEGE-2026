"""Assemble the final Amazon submission with exactly two TSV files."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "Green_Sync_Innovators_submission"
CODE = DEST / "code/business_entity_resolution"


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing folder: {DEST}")

    # The only TSV files in the package are the exact last-submitted pair.
    for name in ["matching_results.tsv", "candidate_pairs.tsv"]:
        copy_file(ROOT / "SUBMIT/S_A95_B" / name, DEST / "output" / name)

    for name in ["ensemble_final.py", "README.md", "requirements.txt", "MODEL_LICENSES.md"]:
        target = CODE / ("src/ensemble_final.py" if name == "ensemble_final.py" else name)
        copy_file(ROOT / "final_package_source" / name, target)
    copy_file(ROOT / "final_package_source/Documentation_template.md",
              DEST / "Documentation_template.md")
    copy_file(ROOT / "DATASET THEY GIVEN/student_resource/utils/validate_submission.py",
              CODE / "src/validate_submission.py")

    # Exact baseline implementation and checkpoints.
    for source in (ROOT / "file1_code/code/business_entity_resolution/src").glob("*.py"):
        if source.name not in {"ab_test.py", "analysis.py", "compare_s2.py", "shift_check.py"}:
            copy_file(source, CODE / "src/baseline" / source.name)
    copy_file(ROOT / "file1_code/code/business_entity_resolution/tests/test_metric.py",
              CODE / "src/baseline/tests/test_metric.py")
    packaged_test = CODE / "src/baseline/tests/test_metric.py"
    packaged_test.write_text(
        packaged_test.read_text(encoding="utf-8").replace(
            'os.path.join(os.path.dirname(__file__), "..", "src")',
            'os.path.join(os.path.dirname(__file__), "..")'
        ),
        encoding="utf-8"
    )
    for source in (ROOT / "file1_code/models").glob("*"):
        if source.is_file():
            copy_file(source, CODE / "models/baseline" / source.name)
    copy_file(ROOT / "file1_code/artifacts/lexicon.json",
              CODE / "models/baseline/lexicon.json")
    copy_file(ROOT / "file1_code/code/business_entity_resolution/requirements.txt",
              CODE / "environments/baseline-requirements.txt")
    copy_file(ROOT / "file1_code/REPRODUCE.md",
              CODE / "environments/BASELINE_REPRODUCE.md")

    # v11/v12 scoring source. Exclude caches, outputs and experimental reports.
    production_scoring = {
        "block.py", "ce.py", "check_submission.py", "config.py", "data.py", "decide.py",
        "eval_blocking.py", "features.py", "infer.py", "io_convert.py", "learn_dict.py",
        "metric.py", "noise_stats.py", "normalize.py", "preprocess.py", "progress.py",
        "run_all.py", "sib.py", "synth.py", "train.py"
    }
    for source in (ROOT / "solution/src").glob("*.py"):
        if source.name in production_scoring:
            copy_file(source, CODE / "src/scoring" / source.name)
    copy_file(ROOT / "solution/requirements.txt",
              CODE / "environments/scoring-requirements.txt")

    # Saved Kaggle launchers preserve the actual v11/v12 run parameters.
    for run in ["ber-t11-train", "ber-c11-infer", "ber-t12-train", "ber-c12-infer"]:
        base = ROOT / "solution/kaggle" / run
        copy_file(base / "main.py", CODE / "src/kaggle_launchers" / run / "main.py")
        copy_file(base / "kernel-metadata.json",
                  CODE / "src/kaggle_launchers" / run / "kernel-metadata.json")

    # Secondary component source and model files; deliberately omit every TSV/output.
    secondary_root = ROOT / "file2_code/code/business_entity_resolution"
    for source in secondary_root.rglob("*"):
        if not source.is_file() or source.suffix not in {".py", ".json", ".pkl", ".txt", ".cbm", ".md"}:
            continue
        if source.name == "Documentation_template.md":
            continue
        relative = source.relative_to(secondary_root)
        copy_file(source, CODE / "src/secondary" / relative)
    copy_file(secondary_root / "requirements.txt",
              CODE / "environments/secondary-requirements-original.txt")

    output_hashes = {
        name: sha256(DEST / "output" / name)
        for name in ["matching_results.tsv", "candidate_pairs.tsv"]
    }
    provenance = {
        "team": "Green Sync Innovators",
        "leaderboard_score": 0.983526,
        "leaderboard_rank_in_supplied_screenshot": 743,
        "last_submitted_source_folder": "SUBMIT/S_A95_B",
        "output_sha256": output_hashes,
        "candidate_construction": {
            "v11_scored_pairs": 8806261,
            "selected_pairs_outside_v11": 35176,
            "unique_candidate_pairs": 8841437,
            "verified_relation": "candidate_pairs = unique(v11 scored pairs UNION final matches)"
        },
        "output_counts": {
            "source1_rows": 1732544,
            "matched_pairs": 5766118,
            "candidate_pairs": 8841437,
            "empty_matching_rows": 101804,
            "empty_candidate_rows": 24399
        },
        "reproduction_note": (
            "The baseline archive input is byte-identical and v11 scores were recovered. "
            "The secondary component supplied after the challenge contains a later output hash, "
            "and the private v12 score artifact was unavailable to this packaging account."
        )
    }
    (CODE / "provenance").mkdir(parents=True, exist_ok=True)
    (CODE / "provenance/audit.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Syntax check packaged Python sources before creating the archive.
    syntax_errors = []
    for source in (CODE / "src").rglob("*.py"):
        try:
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        except SyntaxError as exc:
            syntax_errors.append(f"{source}: {exc}")
    if syntax_errors:
        raise SystemExit("Packaged source has syntax errors:\n" + "\n".join(syntax_errors))

    all_files = [path for path in DEST.rglob("*") if path.is_file()]
    tsv_files = [path.relative_to(DEST).as_posix() for path in all_files if path.suffix.lower() == ".tsv"]
    if sorted(tsv_files) != ["output/candidate_pairs.tsv", "output/matching_results.tsv"]:
        raise SystemExit(f"Unexpected TSV files in package: {tsv_files}")

    manifest = {path.relative_to(DEST).as_posix(): sha256(path) for path in all_files}
    (CODE / "provenance/SHA256SUMS.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    archive = ROOT / "Green_Sync_Innovators_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as out:
        for path in DEST.rglob("*"):
            if path.is_file():
                out.write(path, path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise SystemExit("ZIP integrity test failed")
        names = check.namelist()
        if sorted(name for name in names if name.lower().endswith(".tsv")) != [
            "output/candidate_pairs.tsv", "output/matching_results.tsv"
        ]:
            raise SystemExit("ZIP contains an unexpected TSV")
        for name, expected in output_hashes.items():
            actual = hashlib.sha256(check.read("output/" + name)).hexdigest()
            if actual != expected:
                raise SystemExit(f"Output changed while packaging: {name}")
    print(json.dumps({
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "python_files_checked": len(list((CODE / "src").rglob("*.py"))),
        "tsv_files": tsv_files,
        "output_sha256": output_hashes
    }, indent=2))


if __name__ == "__main__":
    main()
