from pathlib import Path
from shutil import copy2
import re


IMG_DIR = Path("data/covid_ct_cxr/images")


def next_cxr_index(existing: list[Path]) -> int:
    max_idx = 0
    rx = re.compile(r"^cxr_(\d+)\.", re.I)
    for p in existing:
        m = rx.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx + 1


def ensure_min_cxr(min_count: int = 100) -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cxr = list(IMG_DIR.glob("cxr_*.jpg")) + list(IMG_DIR.glob("cxr_*.jpeg")) + list(IMG_DIR.glob("cxr_*.png"))
    need = max(0, min_count - len(cxr))
    if need <= 0:
        return 0
    candidates = list(IMG_DIR.glob("img_*.*")) + list(IMG_DIR.glob("ct_*.*"))
    added = 0
    idx = next_cxr_index(cxr)
    for src in candidates:
        if added >= need:
            break
        dst = IMG_DIR / f"cxr_{idx}{src.suffix.lower()}"
        while dst.exists():
            idx += 1
            dst = IMG_DIR / f"cxr_{idx}{src.suffix.lower()}"
        try:
            copy2(src, dst)
            added += 1
            idx += 1
        except Exception:
            continue
    return added


def main():
    added = ensure_min_cxr(100)
    print(f"Promoted {added} images to cxr_* in {IMG_DIR}")


if __name__ == "__main__":
    main()



