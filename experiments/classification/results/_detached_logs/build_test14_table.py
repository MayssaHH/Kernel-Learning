import csv
import json
from pathlib import Path

root = Path('experiments/classification/results')
out_md = Path('experiments/classification/results_table_test14.md')
out_csv = Path('experiments/classification/results_table_test14.csv')
paper_json = Path('experiments/classification/paper_smkl_2025_results.json')

datasets = [
    'breastcancer',
    'ionosphere',
    'spambase',
    'banknote',
    'haberman',
    'mammographic',
    'parkinsons',
    'wine',
    'iris',
    'heart',
]

paper = {}
if paper_json.exists():
    p = json.loads(paper_json.read_text(encoding='utf-8'))
    for row in p.get('accuracy_percent_table2', []):
        paper[row['dataset']] = row

rows = []
for ds in datasets:
    cands = sorted(root.glob(f'*_test14_uci_{ds}'), key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = None
    manifest = None

    for d in cands:
        m = d / 'manifest.json'
        if not m.exists():
            continue
        try:
            manifest = json.loads(m.read_text(encoding='utf-8'))
            chosen = d
            break
        except Exception:
            continue

    if manifest is None:
        rows.append(
            {
                'dataset': ds,
                'run_uuid': '',
                'protocol': '',
                'our_archi_all_rbf_percent': '',
                'paper_EasyMKL': '',
                'paper_AverageMKL': '',
                'paper_CKA': '',
                'paper_SMKL': '',
                'delta_vs_EasyMKL': '',
                'delta_vs_AverageMKL': '',
                'delta_vs_CKA': '',
                'delta_vs_SMKL': '',
                'results_dir': '',
            }
        )
        continue

    cfg = manifest.get('config', {})
    s = manifest.get('summary', {}).get('all_rbf', {})
    our = s.get('table_accuracy_percent_raw')
    if our is None and s.get('mean_accuracy') is not None:
        our = float(s['mean_accuracy']) * 100.0

    paper_row = paper.get(ds, {})

    def delta(v):
        if our is None or v is None or v == '':
            return ''
        return round(float(our) - float(v), 3)

    rows.append(
        {
            'dataset': ds,
            'run_uuid': manifest.get('run_uuid', ''),
            'protocol': cfg.get('protocol', ''),
            'our_archi_all_rbf_percent': '' if our is None else round(float(our), 3),
            'paper_EasyMKL': paper_row.get('EasyMKL', ''),
            'paper_AverageMKL': paper_row.get('AverageMKL', ''),
            'paper_CKA': paper_row.get('CKA', ''),
            'paper_SMKL': paper_row.get('Algorithm1_SMKL', ''),
            'delta_vs_EasyMKL': delta(paper_row.get('EasyMKL')),
            'delta_vs_AverageMKL': delta(paper_row.get('AverageMKL')),
            'delta_vs_CKA': delta(paper_row.get('CKA')),
            'delta_vs_SMKL': delta(paper_row.get('Algorithm1_SMKL')),
            'results_dir': str(chosen).replace('\\', '/'),
        }
    )

fields = [
    'dataset',
    'run_uuid',
    'protocol',
    'our_archi_all_rbf_percent',
    'paper_EasyMKL',
    'paper_AverageMKL',
    'paper_CKA',
    'paper_SMKL',
    'delta_vs_EasyMKL',
    'delta_vs_AverageMKL',
    'delta_vs_CKA',
    'delta_vs_SMKL',
    'results_dir',
]

with out_csv.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

lines = []
lines.append('# Test14 Final Results Table')
lines.append('')
lines.append('Latest run per dataset (Archi + All RBF), compared with paper Table 2 accuracies (%).')
lines.append('')
lines.append('| Dataset | Our Archi+AllRBF (%) | EasyMKL | AverageMKL | CKA | SMKL | Delta vs Easy | Delta vs Avg | Delta vs CKA | Delta vs SMKL | Run UUID |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for r in rows:
    def fmt(x):
        return '' if x == '' else str(x)

    lines.append(
        '| {dataset} | {our} | {easy} | {avg} | {cka} | {smkl} | {d_easy} | {d_avg} | {d_cka} | {d_smkl} | {uuid} |'.format(
            dataset=r['dataset'],
            our=fmt(r['our_archi_all_rbf_percent']),
            easy=fmt(r['paper_EasyMKL']),
            avg=fmt(r['paper_AverageMKL']),
            cka=fmt(r['paper_CKA']),
            smkl=fmt(r['paper_SMKL']),
            d_easy=fmt(r['delta_vs_EasyMKL']),
            d_avg=fmt(r['delta_vs_AverageMKL']),
            d_cka=fmt(r['delta_vs_CKA']),
            d_smkl=fmt(r['delta_vs_SMKL']),
            uuid=r['run_uuid'],
        )
    )

lines.append('')
lines.append('## Notes')
lines.append('')
lines.append('- Delta columns are `Our - Paper` in percentage points.')
lines.append('- Our values use the latest `*_test14_uci_<dataset>/manifest.json` in `experiments/classification/results`.')
lines.append('- Protocol should be `paper_strict` for fair comparability.')
lines.append('')
lines.append('## Result Paths')
lines.append('')
for r in rows:
    if r['results_dir']:
        lines.append(f"- `{r['dataset']}`: `{r['results_dir']}`")

out_md.write_text('\n'.join(lines), encoding='utf-8')

print('WROTE', out_md)
print('WROTE', out_csv)
print('ROWS', len(rows))
