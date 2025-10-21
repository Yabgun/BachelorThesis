from pathlib import Path
import csv

IMG_DIR = Path('data/covid_ct_cxr/images')
MANIFEST = Path('data/covid_ct_cxr/manifest.csv')

rows = []
for p in sorted(IMG_DIR.glob('*.*')):
    mod = 'ct' if p.name.lower().startswith('ct_') else 'cxr' if p.name.lower().startswith('cxr_') else 'img'
    rows.append({'filepath': str(p), 'modality': mod})

MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['filepath','modality'])
    w.writeheader()
    w.writerows(rows)

print(f'Wrote {len(rows)} rows to {MANIFEST}')