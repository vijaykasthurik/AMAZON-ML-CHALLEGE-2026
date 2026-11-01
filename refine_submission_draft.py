"""Add an independently measured baseline delta and a diagram to the review ZIP."""
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'Green_Sync_Innovators_submission_DRAFT'
CODE = PACKAGE / 'code/business_entity_resolution'


def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    counts = {}
    paths = [ROOT/'Friend_/matching_results (1).tsv',
             ROOT/'SUBMIT/S_A95_B/matching_results.tsv',
             ROOT/'DATASET THEY GIVEN/student_resource/dataset/test/test_source1.tsv']
    with paths[0].open(encoding='utf-8') as a, paths[1].open(encoding='utf-8') as b, paths[2].open(encoding='utf-8') as c:
        for f in [a,b,c]: next(f)
        for row in itertools.zip_longest(a,b,c):
            if any(x is None for x in row): raise ValueError('Row-count mismatch')
            aid, av = row[0].rstrip('\r\n').split('\t')
            bid, bv = row[1].rstrip('\r\n').split('\t')
            fields = row[2].rstrip('\r\n').split('\t')
            if not aid == bid == fields[0]: raise ValueError('Row-order mismatch')
            country = fields[-1]
            stat = counts.setdefault(country, dict(rows=0, changed_rows=0, added_pairs=0, removed_pairs=0))
            aset = set(av.split(',')) if av else set()
            bset = set(bv.split(',')) if bv else set()
            stat['rows'] += 1
            stat['changed_rows'] += aset != bset
            stat['added_pairs'] += len(bset-aset)
            stat['removed_pairs'] += len(aset-bset)
    audit = json.loads((CODE/'audit.json').read_text())
    audit['measured_final_vs_baseline'] = counts
    (CODE/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    doc = PACKAGE/'Documentation_template.md'
    text = doc.read_text(encoding='utf-8').split('\n<!-- measured-pipeline-appendix -->')[0]
    text += '''
<!-- measured-pipeline-appendix -->

## Pipeline Diagram and Exact Submitted Artifact

The final artifact is `SUBMIT/S_A95_B/matching_results.tsv`, paired with the
candidate TSV from the same directory. The package retains these exact bytes.
All components below are parts of one team system; component labels describe
their technical roles, not individual authorship.

```mermaid
flowchart TD
    D[Official training and test data] --> B[Baseline normalisation and blocking]
    B --> L[Two-stage LightGBM and expected-F0.5 decoding]
    L --> BASE[Baseline predictions]
    D --> S[Score-pipeline normalisation and sparse retrieval]
    S --> V11[XGBoost with MiniLM: v11]
    S --> V12[XGBoost with BGE: v12]
    D -. Original secondary pipeline still unresolved .-> SECOND[Secondary predictions]
    BASE --> E[US and India ensemble rules]
    V11 --> E
    V12 --> E
    SECOND --> E
    BASE --> FR[Preserve France predictions]
    E --> FINAL[S_A95_B matching_results.tsv]
    FR --> FINAL
    B -. Candidate provenance reconciliation required .-> CAND[candidate_pairs.tsv]
    S -. Candidate provenance reconciliation required .-> CAND
```

The diagram describes the supplied ensemble account; it is not proof of an
executed end-to-end rebuild. The following changes were independently measured
by comparing the baseline and submitted prediction sets for every Source 1 row.

| Country | Source 1 rows | Changed rows | Added pairs | Removed pairs |
|---|---:|---:|---:|---:|
'''
    for country, stat in sorted(counts.items()):
        text += f"| {country} | {stat['rows']:,} | {stat['changed_rows']:,} | {stat['added_pairs']:,} | {stat['removed_pairs']:,} |\n"
    text += '\nThese differences are measured output changes, not validation errors or accuracy gains.\n'
    doc.write_text(text,encoding='utf-8')
    excluded = {'ab_test.py','compare_s2.py','shift_check.py','analysis.py',
                'benchmark_local_qwen.py','benchmark_tfidf_retrieval.py','neural_trial.py',
                'run_density_stability_check.py','diagnose_step1_step2.py','diagnose_india_misses.py'}
    # Only omit non-production tools from the ZIP; original contributor folders stay intact.
    files = [p for p in PACKAGE.rglob('*') if p.is_file()
             and p.name not in excluded and p.name != 'SHA256SUMS.json'
             and '__pycache__' not in p.parts]
    manifest = {p.relative_to(PACKAGE).as_posix():sha(p) for p in files}
    manifest_path = PACKAGE/'SHA256SUMS.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive = ROOT/'Green_Sync_Innovators_submission_DRAFT.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files+[manifest_path]: z.write(p,p.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in ['matching_results.tsv','candidate_pairs.tsv']:
            assert hashlib.sha256(z.read('output/'+name)).hexdigest() == audit['outputs'][name]['sha256']
    print(json.dumps({'measured_changes':counts,'zip':str(archive),'bytes':archive.stat().st_size},indent=2))


if __name__ == '__main__': main()
