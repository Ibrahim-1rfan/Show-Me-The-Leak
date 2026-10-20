"""Summarize user-provided results without inference or changing saved reports."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_metrics import build_comparison
from benchmark_workflow import MODEL_KEYS, TEXT_RULE_ATTRIBUTES, collect_runs, scope_records, write_json


def csv_table(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


sample_path = next((ROOT / 'Leak_Benchmark_Results').rglob('sample.json'))
sample = json.loads(sample_path.read_text(encoding='utf-8'))
sample_dir = sample_path.parent
latest_qwen_config = sorted((sample_dir / 'qwen3_vl').glob('*/run_config.json'))[-1]
protocol = json.loads(latest_qwen_config.read_text(encoding='utf-8'))
reports, provenance = collect_runs(sample_dir, MODEL_KEYS, protocol['protocol_id'])
taxonomy = sample['taxonomy']
ids = [row['image_id'] for row in sample['images']]
strata = {row['image_id']: row['stratum'] for row in sample['images']}
output = ROOT / 'outputs' / 'benchmark_review'
output.mkdir(parents=True, exist_ok=True)
comparisons = {}
for scope, rows, attributes in [('full_taxonomy', reports, taxonomy),
                                ('shared_text_labels', scope_records(reports, TEXT_RULE_ATTRIBUTES), TEXT_RULE_ATTRIBUTES)]:
    result = build_comparison(rows, attributes, requested_models=MODEL_KEYS, sample_image_ids=ids,
        sample_strata=strata, attribute_to_category={a: t['taxonomy_category'] for a, t in taxonomy.items()})
    comparisons[scope] = result
    write_json(output / f'{scope}.json', result)
    for table in ('summaries', 'coverage', 'per_image', 'per_attribute'):
        csv_table(output / f'{scope}_{table}.csv', result[table])

paired = build_comparison(reports, taxonomy, requested_models=['qwen3_vl', 'qwen25_vl'], sample_image_ids=ids, sample_strata=strata)
trio = build_comparison(reports, taxonomy, requested_models=['qwen3_vl', 'qwen25_vl', 'chandra'], sample_image_ids=ids, sample_strata=strata)
csv_table(output / 'qwen_paired_comparison.csv', paired['summaries'])
csv_table(output / 'qwen_chandra_paired_comparison.csv', trio['summaries'])
write_json(output / 'provenance.json', provenance)

failures = {}
for model in MODEL_KEYS:
    rows = [r for r in reports if r['model_key'] == model]
    failed = [r for r in rows if r['status'] != 'ok']
    reasons = Counter()
    for row in failed:
        prediction = row.get('prediction', {})
        error = prediction.get('error') or row.get('error')
        if prediction.get('attempts'):
            error = prediction['attempts'][-1].get('error', error)
        reasons[error or 'Unspecified error'] += 1
    failures[model] = {'successful_images': len(rows) - len(failed), 'failed_images': len(failed),
                       'failure_reasons': dict(reasons)}
write_json(output / 'failure_summary.json', failures)


def score(value):
    return 'unavailable' if value is None else f'{value:.3f}'


def table(rows):
    lines = ['| Model | Scored images | Precision | Recall | F1 | Missed labels |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        misses = r['missed_count'] if r['n_images'] else 'unavailable'
        lines.append(f"| {r['model_key']} | {r['n_images']} | {score(r['micro_precision'])} | {score(r['micro_recall'])} | {score(r['micro_f1'])} | {misses} |")
    return '\n'.join(lines)


overall = [r for r in comparisons['full_taxonomy']['summaries'] if r['scope'] == 'all_successful' and r['stratum'] == 'all']
paired_rows = [r for r in paired['summaries'] if r['scope'] == 'common_successful' and r['stratum'] == 'all']
trio_rows = [r for r in trio['summaries'] if r['scope'] == 'common_successful' and r['stratum'] == 'all']
document_rows = [r for r in comparisons['full_taxonomy']['summaries'] if r['scope'] == 'all_successful' and r['stratum'] == 'document']
general_rows = [r for r in comparisons['full_taxonomy']['summaries'] if r['scope'] == 'all_successful' and r['stratum'] == 'general']
text_rows = [r for r in comparisons['shared_text_labels']['summaries'] if r['scope'] == 'all_successful' and r['stratum'] == 'all']
report = f'''# Review of the saved benchmark run

These are measured results from the latest saved runs in `Leak_Benchmark_Results`, not a new GPU run.
The sample contains 50 images: 25 document and 25 general. Source runs and their original protocol
IDs are listed in `provenance.json`. No archived outputs were modified.

## Full-taxonomy results on each model's successful images

{table(overall)}

**Do not rank these rows directly:** their image coverage differs. Failures are excluded from
quality scores and counted separately. Llama has only five valid outputs, none on documents.
Dots and Florence never reached successful inference; they have no quality scores.

## Paired Qwen comparison: {len(paired['common_image_ids'])} identical successful images

{table(paired_rows)}

## Paired Qwen/Chandra comparison: {len(trio['common_image_ids'])} identical successful images

{table(trio_rows)}

Chandra is an OCR-plus-rule pipeline. The general VLMs directly predict visual privacy labels;
the full-taxonomy scores include non-textual attributes that the OCR rules cannot detect.

## Shared text-label scope (per-model successful images)

{table(text_rows)}

## Document images (per-model successful images)

{table(document_rows)}

## General images (per-model successful images)

{table(general_rows)}

## Failure diagnosis and repair

- **Dots:** the recorded error says Transformers 4.57.1 was installed, while its adapter needs
  4.56.1. It stopped at setup for all 50 images. The saved L4 GPU is hardware-compatible.
- **Florence:** the first run found Transformers 4.57.1 instead of 4.41.2. Later attempts failed
  during imports after installing 4.41.2 with `--no-deps` in Python 3.13. The generic wrapper
  concealed the nested import error, so its exact message is not recoverable from these files.
  Transformers 4.41.2 requires tokenizers >=0.19,<0.20; installing the package alone is insufficient.
- **Llama:** 45/50 outputs failed validation. Representative outputs are natural-language image
  descriptions rather than the requested JSON. These are not legitimate empty PII predictions.
  The dependency repair does not claim to solve this instruction-following problem.
- **Qwen2.5:** five images failed schema/evidence validation. Qwen3 completed all 50 after its
  configured retries.
- **Chandra:** four images failed; saved metadata shows incomplete generation at the token cap.
  Raising its cap would require a separately disclosed protocol/run, not editing these scores.

The repaired notebook uses isolated Python 3.11 model workers and checks imports, versions, and
GPU availability before inference. It keeps complete dependency resolution and retains nested
import errors. The notebook can include older equal-setting runs explicitly, with code differences
disclosed in provenance, so only the failed profiles need to be retried.

## Interpretation limits

Precision is agreement with the supplied image-level labels, which may be incomplete. Missed
counts are image/attribute occurrences, not individual physical instances. Raw privacy values
are deliberately omitted from this report. Per-image/per-attribute omissions are in the CSVs.
There are no reference boxes or transcripts here, so IoU, ROUGE-L, CER, and WER remain unavailable.
No six-model paired ranking is possible until dots and Florence have valid outputs.

Sources: [Transformers 4.41.2 dependency requirements](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/dependency_versions_table.py),
[uv Python environment management](https://docs.astral.sh/uv/guides/install-python/).
'''
(output / 'README.md').write_text(report, encoding='utf-8')
print(table(overall))
print('\nPAIRED QWEN:', len(paired['common_image_ids']))
print(table(paired_rows))
print('\nREPORT:', output / 'README.md')
