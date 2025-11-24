import json
from pathlib import Path

IN_JSON = Path("data/covid_ct_cxr/ckks_report.json")
OUT_MD = Path("data/covid_ct_cxr/ckks_eval.md")


def main():
    if not IN_JSON.exists():
        raise SystemExit(f"ckks_report.json not found: {IN_JSON}")
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))

    lines = []
    lines.append("# CKKS Homomorfik Şifreleme Değerlendirme Raporu")
    lines.append("")
    lines.append(f"Örnek sayısı: {data.get('samples')}")
    lines.append(f"Mean abs error: {data.get('mean_abs_error')}")
    lines.append(f"Max abs error: {data.get('max_abs_error')}")
    perf = data.get('performance_ms', {})
    lines.append(f"Mean CKKS latency (ms): {perf.get('mean_ckks_ms')}")
    lines.append(f"Mean plaintext latency (ms): {perf.get('mean_plain_ms')}")
    lines.append("")

    pol = data.get('policy', {})
    lines.append("## Politika")
    lines.append(f"top_k: {pol.get('top_k')}")
    lines.append(f"encrypt_columns: {', '.join(pol.get('encrypt_columns', []))}")
    weights = pol.get('weights', {})
    bias = pol.get('bias', 0)
    lines.append("## Ağırlıklar ve bias")
    for k, v in weights.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- bias: {bias}")
    lines.append("")

    lines.append("Not: CKKS yaklaşık değerler üretir; küçük farklar beklenir ve demoda kabul edilebilir.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"CKKS evaluation report -> {OUT_MD}")


if __name__ == '__main__':
    main()