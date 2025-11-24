import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("AES-128-CBC", Path("nist_vectors_aes.py")),
    ("AES-128-ECB", Path("nist_vectors_aes_ecb.py")),
    ("AES-128-CTR", Path("nist_vectors_aes_ctr.py")),
    ("AES-128-GCM", Path("nist_vectors_aes_gcm_crypto.py")),
    ("SHA-256", Path("nist_vectors_sha256.py")),
]

OUT_MD = Path("data/covid_ct_cxr/crypto_eval.md")


def run_script(name, path):
    try:
        res = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=False)
        out = (res.stdout or "") + (res.stderr or "")
        status = "PASS" if "PASS" in out and "FAIL" not in out else "FAIL"
        return name, status, out.strip()
    except Exception as e:
        return name, "ERROR", str(e)


def main():
    lines = ["# NIST/Standart Kriptografi Testleri Sonuçları\n"]
    for name, path in SCRIPTS:
        n, status, out = run_script(name, path)
        lines.append(f"## {n}")
        lines.append(f"Durum: {status}")
        lines.append("Çıktı:")
        lines.append("```")
        lines.append(out)
        lines.append("```")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Crypto evaluation report -> {OUT_MD}")


if __name__ == '__main__':
    main()