"""Build an explicitly incomplete audit package without changing submitted outputs."""
from pathlib import Path
import ast
import base64
import hashlib
import io
import json
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'Green_Sync_Innovators_submission_DRAFT'
CODE = DEST / 'code/business_entity_resolution'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(path):
    rows = pairs = empty = maximum = 0
    with path.open(encoding='utf-8') as f:
        header = next(f).rstrip('\n')
        for line in f:
            _, values = line.rstrip('\r\n').split('\t')
            n = values.count(',') + 1 if values else 0
            rows += 1
            pairs += n
            empty += not n
            maximum = max(maximum, n)
    return dict(rows=rows, pairs=pairs, empty_rows=empty, max_list_length=maximum,
                mean_list_length=pairs / rows, sha256=digest(path), header=header)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    DEST.mkdir(exist_ok=True)
    audit = {'status': 'INCOMPLETE_NOT_VERIFIED_FOR_UPLOAD', 'outputs': {}}
    for name in ['matching_results.tsv', 'candidate_pairs.tsv']:
        src = ROOT / 'SUBMIT/S_A95_B' / name
        audit['outputs'][name] = stats(src)
        copy(src, DEST / 'output' / name)
    audit['file1'] = stats(ROOT / 'file1_code/output/matching_results.tsv')
    audit['file1_candidates'] = stats(ROOT / 'file1_code/output/candidate_pairs.tsv')
    audit['file1']['identical_to_original'] = audit['file1']['sha256'] == digest(ROOT / 'Friend_/matching_results (1).tsv')
    audit['file2_original_sha256'] = digest(ROOT / 'Friend_/matching_results (2).tsv')
    audit['file2_delivered_sha256'] = digest(ROOT / 'file2_code/output/matching_results.tsv')
    audit['file2_changed_match_lists'] = 500678
    audit['file2_comparison_note'] = '1,732,544 shared unique S1 IDs; comparison sorted each comma-separated match list before equality.'
    target_count = 0
    for source in [2, 3]:
        p = ROOT / f'DATASET THEY GIVEN/student_resource/dataset/test/test_source{source}.tsv'
        with p.open('rb') as f:
            target_count += sum(1 for _ in f) - 1
    all_pairs = audit['outputs']['matching_results.tsv']['rows'] * target_count
    audit['comparison_space'] = dict(test_targets=target_count, all_country_cross_product=all_pairs,
        reduction_ratio=1 - audit['outputs']['candidate_pairs.tsv']['pairs'] / all_pairs,
        caveat='Arithmetic reduction of the preserved candidate file only, including cross-country pairs in denominator; not a verified reduction of the full ensemble inference set.')
    recovered = ROOT / 'audit_recovery/c11/outputs/test_scores.parquet'
    if recovered.exists() and recovered.stat().st_size:
        audit['v11_recovered_scores'] = dict(bytes=recovered.stat().st_size, sha256=digest(recovered),
            source='vijaykasthurik/ber-c11-infer', note='Downloaded separately to audit_recovery; not yet used to reproduce the final ensemble.')
        log = ROOT / 'audit_recovery/c11/ber-c11-infer.log'
        if log.exists(): copy(log, CODE / 'component_notes/notebooks/ber-c11-infer.log')
    for label in ['file1', 'file2']:
        source = ROOT / (label + '_code') / 'code/business_entity_resolution'
        for p in source.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            rel = p.relative_to(source)
            if p.suffix == '.py':
                # Keep each component isolated; preserve file2 root helper imports.
                target = CODE / 'src' / label / rel
            elif 'models' in rel.parts:
                target = CODE / 'src' / label / rel
            else:
                target = CODE / 'component_notes' / label / rel
            copy(p, target)
    for p in (ROOT / 'file1_code/models').iterdir():
        copy(p, CODE / 'models/file1' / p.name)
    copy(ROOT / 'file1_code/artifacts/lexicon.json', CODE / 'models/file1/lexicon.json')
    copy(ROOT / 'file1_code/REPRODUCE.md', CODE / 'component_notes/file1/REPRODUCE_original.md')
    for p in (ROOT / 'solution/src').glob('*.py'):
        copy(p, CODE / 'src/score_pipeline_current' / p.name)
    copy(ROOT / 'solution/requirements.txt', CODE / 'component_notes/score_pipeline_requirements_original.txt')
    for run in ['ber-t11-train', 'ber-c11-infer', 'ber-t12-train', 'ber-c12-infer']:
        source = ROOT / 'solution/kaggle' / run / 'main.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in n.targets))
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(value))) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.py'):
                    parts = Path(member.name).parts
                    if '..' in parts or Path(member.name).is_absolute():
                        raise ValueError('Unsafe embedded source path')
                    target = CODE / 'src/notebook_snapshots' / run / Path(member.name).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        copy(source, CODE / 'src/notebook_launchers' / run / 'main.py')
        copy(source.with_name('kernel-metadata.json'), CODE / 'component_notes/notebooks' / (run + '.json'))
    copy(ROOT / 'DATASET THEY GIVEN/student_resource/utils/validate_submission.py', CODE / 'src/validate_submission.py')
    copy(ROOT / 'Friend_/matching_results (1).tsv', CODE / 'reference_outputs/file1_matching_results.tsv')
    copy(ROOT / 'Friend_/matching_results (2).tsv', CODE / 'reference_outputs/file2_matching_results.tsv')
    audit['blockers'] = [
        'Delivered file2 code/output has not been linked to the original file2 predictions.',
        'Private dinnypaulnavis/ber-c12-infer access denied; v12 score artifacts not recovered.',
        'Exact ensemble blend, score-column choice, missing-score behavior and tie/order behavior are not yet established from executable provenance.',
        'Submitted candidate file must be reconciled with the actual candidate sets scored by all contributing components.',
        'No full raw-data-to-final-output reproduction has been completed; dependency environments differ across components.'
    ]
    (CODE / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (CODE / 'requirements.txt').write_text(
        '# File1 component only. This is NOT a complete environment for the final ensemble.\n'
        '-r component_notes/file1/requirements.txt\n', encoding='utf-8')
    (CODE / 'README.md').write_text('''# Reproduction status: INCOMPLETE DRAFT

This package preserves the submitted S_A95_B outputs and the currently available
component code. It is not yet a verified end-to-end reproduction package.
The two TSVs in output/ are unchanged copies, not newly regenerated predictions.
The reference_outputs/ files are provenance evidence, not raw-data substitutes.

## Components

- src/file1/src/: two-stage LightGBM baseline, with exact delivered models under models/file1/.
- src/file2/: delivered LightGBM/CatBoost code; its supplied predictions differ from the required original file2.
- src/score_pipeline_current/: local v11/v12 scoring development source.
- src/notebook_snapshots/: source decoded from the saved v11/v12 training and inference launchers.
- src/notebook_launchers/: original Kaggle launchers, preserving run settings and dependencies.
- component_notes/: original dependency files, README files and notebook metadata.
- audit.json: measured output statistics, fingerprints and unresolved reproduction blockers.

## Available baseline reproduction command

Use the original Python 3.10 environment specified in component_notes/file1/requirements.txt.
Install those pinned dependencies in a separate environment. From this code folder:

    python src/reproduce_file1.py --data-dir /path/to/dataset --work-dir /path/to/scratch/file1

The official dataset folder must contain train/ and test/. The command regenerates
normalised data and candidates, then predicts with the supplied baseline checkpoints.
It checks BOTH regenerated output hashes against the delivered file1 manifest.
It does not recreate S_A95_B and has not been run to completion in this audit.
Raw training data is needed to regenerate the lexicon. Allow substantial RAM, disk
space and runtime; the component's original guide estimates several hours for full retraining.

For training from scratch, follow component_notes/file1/REPRODUCE_original.md using
src/file1/src/. Models and lexicon were supplied by the contributor; their training
provenance and validation results have not been independently rerun here.

## Final-output validation (format only)

    python src/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir /path/to/dataset/test --check-ids

The original S_A95_B files previously passed the supplied validator including ID
existence. Format validation does not verify model provenance or candidate completeness.

## What remains before a final ZIP

Resolve every blocker in audit.json; obtain the exact file2 pipeline; recover v12
scores and exact ensemble configuration; reconcile the candidate audit set; then
run the complete chain in isolated environments and compare both final SHA-256 hashes.
Do not use reference predictions or an output-copy operation as a claim of regeneration.
''', encoding='utf-8')
    (CODE / 'src/reproduce_file1.py').write_text('''"""Reproduce only file1 using supplied checkpoints; assert both output hashes."""
import argparse, hashlib, os, pathlib, shutil, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--work-dir', required=True)
    a = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    work = pathlib.Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise SystemExit('Use an empty work directory to prevent stale-cache contamination.')
    env = dict(os.environ, BER_DATA_DIR=str(pathlib.Path(a.data_dir).resolve()),
               BER_WORK_DIR=str(work), BER_OUTPUT_DIR=str(work/'output'))
    src = root/'src/file1/src'
    for args in [('preprocess.py',), ('candidates.py','test')]:
        subprocess.run([sys.executable,*args],cwd=src,env=env,check=True)
    (work/'models').mkdir(exist_ok=True)
    for p in (root/'models/file1').iterdir():
        if p.name != 'lexicon.json': shutil.copy2(p,work/'models'/p.name)
    subprocess.run([sys.executable,'infer.py'],cwd=src,env=env,check=True)
    expected={'matching_results.tsv':'bf9d90aaddb06c5cda7191d6f73ea8ca5192d00746e6f73f1f5ae40a6b367fa8',
              'candidate_pairs.tsv':'d10ad1635f11c8c0c5581a33a79bc743762b5d239eb2b852e54c937697f6cf56'}
    for name,want in expected.items():
        with (work/'output'/name).open('rb') as f:
            h=hashlib.sha256()
            for b in iter(lambda:f.read(1048576),b''): h.update(b)
        if h.hexdigest()!=want: raise SystemExit(name+' differs: '+h.hexdigest())
        print(name+': byte-identical')

if __name__=='__main__': main()
''', encoding='utf-8')
    m = audit['outputs']['matching_results.tsv']
    c = audit['outputs']['candidate_pairs.tsv']
    fc = audit['file1_candidates']
    (DEST / 'Documentation_template.md').write_text(f'''# Business Entity Resolution — Green Sync Innovators

**Status: working methodology draft; reproduction is incomplete.**

Team members: Vijay Kasthuri K; Sudharsan E; AKASH SM; Dinny Paul Navis C.
The supplied leaderboard screenshot records score 0.983526 and rank 743.
These are screenshot observations, not independently recomputed test metrics.

## 1. Executive Summary

Our submitted result combines a two-stage LightGBM baseline with evidence from
two scoring variants and a second team prediction file. The final selection
preserves the baseline's France predictions and applies conservative changes
to US and India. This draft distinguishes confirmed code and artifacts from
the parts whose exact reproduction is still being reconciled.

## 2. Methodology

The task links each Source 1 business to zero or more Source 2/3 records.
Names, addresses, legal forms and scripts vary across sources. We normalise
these fields, retrieve plausible pairs, score those pairs and resolve target
ownership. The objective is per-Source-1 macro F0.5, including singleton rows.

The supplied account of the final ensemble is:
1. Start with file1 (reported leaderboard score 0.982907).
2. Remove US/India links whose v11 score is below 0.05.
3. Remove US/India links absent from file2 when the v11/v12 blend is below 0.30.
4. Add US/India links from file2 when both variant scores are at least 0.95,
   subject to a single Source 1 owner per target.
5. Leave France as in file1.

This is the supplied recipe, not yet an executed reproduction. Blend weights,
score-column selection, absent-score behavior and conflict ordering still need
executable confirmation. We will not replace these details with guesses.

## 3. Candidate Generation / Blocking

The verified file1 source retrieves Source 1 candidates for each Source 2/3
record within country. It combines character trigrams and composite word keys,
retrieves six top-10 lists, rescores and retains pairs reaching the top three
of any list. The matching models use K_MODEL=2. Normalisation includes a
training-derived native-script lexicon and a rule-based romanisation fallback.

The local v11/v12 source uses two-way IDF-weighted sparse retrieval with phonetic
name, address, numeric and composite tokens, followed by learned pruning.
The saved launchers preserve the run-specific settings. These settings must be
used rather than assuming the current source defaults reproduce historical runs.

The preserved final candidate TSV contains **{c['pairs']:,} pairs** across
**{c['rows']:,} Source 1 rows**, averaging **{c['mean_list_length']:.6f}** candidates
per row (maximum {c['max_list_length']:,}). The delivered file1 candidate output
contains **{fc['pairs']:,} pairs**. Their different sizes require reconciliation:
the final candidate audit must faithfully represent the relevant scored pairs
across contributing pipelines. Containment of final matches alone is insufficient.
Against the unrestricted cross product of {m['rows']:,} Source 1 records and
{target_count:,} target records, this preserved file has an arithmetic reduction
of {audit['comparison_space']['reduction_ratio'] * 100:.8f}%. The denominator
includes cross-country pairs. This is not a verified reduction for the full
ensemble: its actual inference candidate set remains unresolved. No final
blocking-recall claim is made in this draft.

## 4. Model Architecture and Feature Engineering

File1 uses two LightGBM stage-1 fold models with 115 features and one stage-2
LightGBM model with 36 features. All three loaded successfully, and their feature
names exactly matched the supplied metadata. Pair features cover fuzzy name and
address agreement, token alignment, numeric gaps, legal forms, acronyms, rarity,
missing fields and retrieval context. Stage 2 incorporates stage-1 scores and
candidate-graph competition. The recorded decoder is expected-F0.5 `ef_0.1`.

The v11/v12 launchers configure five-fold XGBoost pair and contextual models,
with cross-encoder scores: v11 defaults to cross-encoder/ms-marco-MiniLM-L6-v2;
v12 specifies BAAI/bge-reranker-base. The v11 launchers set BER_HOLD=80; v12 sets
BER_HOLD=100, which the training source treats as out-of-fold evaluation rather
than an untouched holdout. The optional France adaptation is separate from the
reported final ensemble, which retains file1's France predictions.

The delivered file2 README instead describes a 70/30 LightGBM/CatBoost blend.
Its included predictions differ from the original file2 in 500,678 Source 1
rows. Another included script references Kaggle scores in directories absent
from the package. The actual original file2 architecture is therefore unresolved.
Model licenses and exact model revisions must be verified for every component
once its role in the submitted result is established.

## 5. Results and Error Analysis

The preserved final matching output has **{m['rows']:,} rows**, **{m['pairs']:,}
matched pairs**, and **{m['empty_rows']:,} empty predictions**. The official
validator previously passed both final TSVs with ID-existence checks enabled.
This is a format check, not an accuracy or full reproducibility result.

File1's metadata records stage-1 validation macro F0.5 of 0.9865272000191707
and stage-2 0.9874423118014127. Its guide describes a 4% validation slice of
88,408 Source 1 businesses. These are supplied component results, not metrics
for the final ensemble, and have not been independently recomputed here.

Confusable names, shared addresses, numeric disagreements, missing addresses
and cross-script variation are feature-design concerns visible in the code.
No measured final-ensemble false-positive/false-negative counts are available;
we do not present assumed error patterns as a completed error study.

## 6. Conclusion

The available artifacts establish the exact baseline prediction file and preserve
the final leaderboard outputs. Completing the audit requires the original file2
pipeline, accessible v12 artifacts, exact ensemble settings and candidate-set
reconciliation, followed by a raw-data reproduction and output hash comparison.

## Appendix: Code Artifacts

`code/business_entity_resolution/README.md` documents the available baseline
reproduction entry point, source layout and limitations. `audit.json` records
counts, fingerprints and remaining blockers. `output/` contains unchanged copies
of S_A95_B. Reference predictions are retained as evidence and do not substitute
for generating results from the raw data. This archive is a draft, not a claim
that the mandatory reproducibility requirement has been met.
''', encoding='utf-8')
    manifest = {}
    for p in DEST.rglob('*'):
        if p.is_file() and p.name != 'SHA256SUMS.json':
            manifest[p.relative_to(DEST).as_posix()] = digest(p)
    (DEST / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = ROOT / 'Green_Sync_Innovators_submission_DRAFT.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for p in DEST.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in ['matching_results.tsv','candidate_pairs.tsv']:
            assert hashlib.sha256(z.read('output/'+name)).hexdigest() == audit['outputs'][name]['sha256']
    print(json.dumps({'archive':str(archive), 'bytes':archive.stat().st_size,
                      'outputs':audit['outputs'], 'file1_candidates':fc,
                      'file1_exact':audit['file1']['identical_to_original'],
                      'status':audit['status']}, indent=2))


if __name__ == '__main__':
    main()
