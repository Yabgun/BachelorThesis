import json
from pathlib import Path

REPORT_JSON = Path('data/covid_ct_cxr/he_report.json')
OUT_MD = Path('reports/he_eval.md')


def main():
    data = json.loads(REPORT_JSON.read_text(encoding='utf-8'))
    md = []
    md.append('# HE Evaluation Summary')
    md.append('')
    md.append('## Overview')
    md.append('- Samples: %d' % data.get('samples', 0))
    md.append('- Policy top_k: %s' % data.get('policy', {}).get('top_k'))
    md.append('- Encrypt columns: %s' % ', '.join(data.get('policy', {}).get('encrypt_columns', [])))
    md.append('')
    md.append('## Weights and Bias')
    md.append('')
    for k, v in data.get('policy', {}).get('weights', {}).items():
        md.append('- %s: %s' % (k, v))
    md.append('- bias: %s' % data.get('policy', {}).get('bias'))
    md.append('')
    md.append('## Accuracy')
    md.append('- Mean absolute error (plaintext vs HE): %.6f' % float(data.get('mean_abs_error', 0.0)))
    md.append('- Max absolute error: %.6f' % float(data.get('max_abs_error', data.get('mean_abs_error', 0.0))))
    md.append('')
    md.append('## Performance')
    md.append('- Mean HE latency (ms): %.3f' % float(data.get('mean_he_ms', 0.0)))
    md.append('- Mean plaintext latency (ms): %.3f' % float(data.get('mean_plain_ms', 0.0)))
    md.append('')
    md.append('## Notes')
    md.append(data.get('policy', {}).get('notes', ''))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(md), encoding='utf-8')
    print(f'Wrote {OUT_MD}')


if __name__ == '__main__':
    main()