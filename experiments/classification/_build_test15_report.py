import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path('experiments/classification/results')
OUT_MD = Path('experiments/classification/results_test15.md')
OUT_TABLE_MD = Path('experiments/classification/results_table_test15.md')
OUT_CSV = Path('experiments/classification/results_table_test15.csv')
OUT_HEATMAP = Path('experiments/classification/results_test15_delta_heatmap.png')
PAPER_JSON = Path('experiments/classification/paper_smkl_2025_results.json')

MODEL_ORDER = [
    'archi_all_rbf',
    'archi_half_rbf_half_linear',
    'archi_all_linear',
    'full_vector_rbf',
    'full_vector_linear',
]
MODEL_LABEL = {
    'archi_all_rbf': 'Archi + All RBF',
    'archi_half_rbf_half_linear': 'Archi + Half RBF/Half Linear',
    'archi_all_linear': 'Archi + All Linear',
    'full_vector_rbf': 'Full-Vector RBF',
    'full_vector_linear': 'Full-Vector Linear',
}


def parse_dt(s: str):
    if not s:
        return datetime.min
    s = s.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min


def load_latest_test15_manifests():
    latest = {}
    for d in RESULTS_ROOT.glob('*_test15_alpha_pruning_*'):
        mf = d / 'manifest.json'
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding='utf-8'))
        except Exception:
            continue
        cfg = m.get('config', {})
        if cfg.get('experiment_name') != 'test15_alpha_pruning_two_pass':
            continue
        ds = cfg.get('dataset_name')
        if not ds:
            continue
        created = parse_dt(m.get('created_utc'))
        prev = latest.get(ds)
        if prev is None or created > prev['created']:
            latest[ds] = {'manifest': m, 'folder': d.name, 'created': created}
    return latest


def load_paper_accuracy():
    if not PAPER_JSON.exists():
        return {}
    try:
        payload = json.loads(PAPER_JSON.read_text(encoding='utf-8'))
    except Exception:
        return {}
    out = {}
    for row in payload.get('accuracy_percent_table2', []):
        ds = row.get('dataset')
        if ds:
            out[ds] = row
    return out


latest = load_latest_test15_manifests()
paper_acc = load_paper_accuracy()

if not latest:
    raise SystemExit('No test15 manifests found.')

datasets = sorted(latest.keys())

long_rows = []
for ds in datasets:
    m = latest[ds]['manifest']
    summary = m.get('summary', {})
    models = summary.get('models', {})
    prune = summary.get('feature_pruning', {})
    selected_count = float(prune.get('mean_selected_feature_count', float('nan')))
    selected_ratio = float(prune.get('mean_selected_feature_ratio', float('nan')))
    selected_mass = float(prune.get('mean_selected_feature_mass', float('nan')))
    run_uuid = m.get('run_uuid', '')
    for key in MODEL_ORDER:
        if key not in models:
            continue
        entry = models[key]
        pass1 = float(entry.get('pass1_accuracy_percent_raw', float('nan')))
        pass2 = float(entry.get('pass2_accuracy_percent_raw', float('nan')))
        delta = float(entry.get('delta_percent_points', float('nan')))
        long_rows.append(
            {
                'dataset': ds,
                'model_key': key,
                'model_label': MODEL_LABEL[key],
                'pass1_percent': pass1,
                'pass2_percent': pass2,
                'delta_percent_points': delta,
                'selected_features_mean': selected_count,
                'selected_ratio_mean': selected_ratio,
                'selected_mass_mean': selected_mass,
                'run_uuid': run_uuid,
                'result_folder': latest[ds]['folder'],
            }
        )

agg = {}
for key in MODEL_ORDER:
    vals = [r for r in long_rows if r['model_key'] == key]
    if not vals:
        continue
    p1 = np.array([r['pass1_percent'] for r in vals], dtype=np.float64)
    p2 = np.array([r['pass2_percent'] for r in vals], dtype=np.float64)
    d = np.array([r['delta_percent_points'] for r in vals], dtype=np.float64)
    agg[key] = {
        'model_label': MODEL_LABEL[key],
        'mean_pass1_percent': float(np.nanmean(p1)),
        'mean_pass2_percent': float(np.nanmean(p2)),
        'mean_delta_pp': float(np.nanmean(d)),
        'median_delta_pp': float(np.nanmedian(d)),
        'improved_count': int(np.sum(d > 1e-12)),
        'degraded_count': int(np.sum(d < -1e-12)),
        'unchanged_count': int(np.sum(np.abs(d) <= 1e-12)),
    }

# Heatmap: dataset x model deltas
heat = np.full((len(datasets), len(MODEL_ORDER)), np.nan, dtype=np.float64)
for i, ds in enumerate(datasets):
    for j, key in enumerate(MODEL_ORDER):
        row = next((r for r in long_rows if r['dataset'] == ds and r['model_key'] == key), None)
        if row is not None:
            heat[i, j] = row['delta_percent_points']

fig_w = max(10, 1.8 * len(MODEL_ORDER))
fig_h = max(6, 0.6 * len(datasets) + 2.5)
plt.figure(figsize=(fig_w, fig_h))
finite_vals = heat[np.isfinite(heat)]
if finite_vals.size > 0:
    vmax = float(np.nanmax(np.abs(finite_vals)))
    vmax = max(vmax, 1.0)
else:
    vmax = 1.0
im = plt.imshow(heat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, label='Delta Accuracy (Pass2 - Pass1, pp)')
plt.xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABEL[k] for k in MODEL_ORDER], rotation=20, ha='right')
plt.yticks(np.arange(len(datasets)), datasets)
plt.title('Test15 Delta Heatmap (pp)')
for i in range(len(datasets)):
    for j in range(len(MODEL_ORDER)):
        val = heat[i, j]
        if np.isfinite(val):
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color='black')
plt.tight_layout()
OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_HEATMAP, dpi=220)
plt.close()

# CSV export
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow([
        'dataset',
        'model_key',
        'model_label',
        'pass1_percent',
        'pass2_percent',
        'delta_percent_points',
        'selected_features_mean',
        'selected_ratio_mean',
        'selected_mass_mean',
        'run_uuid',
        'result_folder',
    ])
    for r in long_rows:
        w.writerow([
            r['dataset'],
            r['model_key'],
            r['model_label'],
            f"{r['pass1_percent']:.6f}",
            f"{r['pass2_percent']:.6f}",
            f"{r['delta_percent_points']:.6f}",
            f"{r['selected_features_mean']:.6f}",
            f"{r['selected_ratio_mean']:.6f}",
            f"{r['selected_mass_mean']:.6f}",
            r['run_uuid'],
            r['result_folder'],
        ])

# Short table md
short_lines = []
short_lines.append('# Test15 Results Table')
short_lines.append('')
short_lines.append('Latest run per dataset for `test15_alpha_pruning_two_pass`.')
short_lines.append('')
short_lines.append('| Model | Mean Pass1 (%) | Mean Pass2 (%) | Mean Delta (pp) | Improved | Degraded | Unchanged |')
short_lines.append('|---|---:|---:|---:|---:|---:|---:|')
for key in MODEL_ORDER:
    a = agg[key]
    short_lines.append(
        f"| {a['model_label']} | {a['mean_pass1_percent']:.3f} | {a['mean_pass2_percent']:.3f} | {a['mean_delta_pp']:.3f} | {a['improved_count']} | {a['degraded_count']} | {a['unchanged_count']} |"
    )
short_lines.append('')
short_lines.append(f"Heatmap: `{OUT_HEATMAP.as_posix()}`")
OUT_TABLE_MD.write_text('\n'.join(short_lines) + '\n', encoding='utf-8')

# Full explanatory md
lines = []
lines.append('# Test15 Full Report')
lines.append('')
lines.append('This report summarizes the completed `test15_alpha_pruning_two_pass` sweep across the 10 UCI datasets.')
lines.append('')
lines.append('## What Test15 Does')
lines.append('')
lines.append('1. Pass 1 trains models on all original features.')
lines.append('2. A feature mask is built from the source alpha vector (`alpha_source=all_rbf`) using cumulative alpha mass (`keep_mass=0.95`).')
lines.append('3. Pass 2 retrains the same local models on the reduced feature set.')
lines.append('4. Delta is reported as `Pass2 - Pass1` in percentage points.')
lines.append('')
lines.append('Paper methods (EasyMKL, AverageMKL, CKA, SMKL) are still reference-only in Test15; they are not retrained here.')
lines.append('')
lines.append('## Aggregate Model Summary')
lines.append('')
lines.append('| Model | Mean Pass1 (%) | Mean Pass2 (%) | Mean Delta (pp) | Median Delta (pp) | Improved | Degraded | Unchanged |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
for key in MODEL_ORDER:
    a = agg[key]
    lines.append(
        f"| {a['model_label']} | {a['mean_pass1_percent']:.3f} | {a['mean_pass2_percent']:.3f} | {a['mean_delta_pp']:.3f} | {a['median_delta_pp']:.3f} | {a['improved_count']} | {a['degraded_count']} | {a['unchanged_count']} |"
    )

lines.append('')
lines.append('## Per Dataset / Per Model Results')
lines.append('')
lines.append('| Dataset | Model | Pass1 (%) | Pass2 (%) | Delta (pp) | Selected Features (mean) | Selected Ratio (mean) | Selected Mass (mean) | Run UUID |')
lines.append('|---|---|---:|---:|---:|---:|---:|---:|---|')
for r in sorted(long_rows, key=lambda x: (x['dataset'], MODEL_ORDER.index(x['model_key']))):
    lines.append(
        f"| {r['dataset']} | {r['model_label']} | {r['pass1_percent']:.3f} | {r['pass2_percent']:.3f} | {r['delta_percent_points']:.3f} | {r['selected_features_mean']:.2f} | {r['selected_ratio_mean']:.3f} | {r['selected_mass_mean']:.3f} | {r['run_uuid']} |"
    )

lines.append('')
lines.append('## Paper Table-2 Reference (Context Only)')
lines.append('')
lines.append('| Dataset | EasyMKL | AverageMKL | CKA | SMKL |')
lines.append('|---|---:|---:|---:|---:|')
for ds in datasets:
    row = paper_acc.get(ds)
    if row is None:
        lines.append(f'| {ds} | NA | NA | NA | NA |')
        continue
    lines.append(
        f"| {ds} | {float(row['EasyMKL']):.1f} | {float(row['AverageMKL']):.1f} | {float(row['CKA']):.1f} | {float(row['Algorithm1_SMKL']):.1f} |"
    )

lines.append('')
lines.append('## Artifacts')
lines.append('')
lines.append(f"- CSV table: `{OUT_CSV.as_posix()}`")
lines.append(f"- Delta heatmap: `{OUT_HEATMAP.as_posix()}`")
lines.append(f"- Compact summary: `{OUT_TABLE_MD.as_posix()}`")

OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(f'wrote {OUT_MD}')
print(f'wrote {OUT_TABLE_MD}')
print(f'wrote {OUT_CSV}')
print(f'wrote {OUT_HEATMAP}')
print(f'datasets={len(datasets)}, rows={len(long_rows)}')
