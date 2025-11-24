import argparse
from pathlib import Path
from shutil import copy2
import kagglehub


def list_images(root: Path) -> list[Path]:
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            imgs.append(p)
    return imgs


def ensure_min_cxr_images(target_dir: Path, min_count: int) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in target_dir.glob("cxr_*.jpg")] + [p for p in target_dir.glob("cxr_*.jpeg")] + [p for p in target_dir.glob("cxr_*.png")]
    need = max(0, min_count - len(existing))
    if need <= 0:
        return 0
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    src_root = Path(path)
    candidates = list_images(src_root)
    copied = 0
    i = 0
    while copied < need and i < len(candidates):
        src = candidates[i]
        i += 1
        out = target_dir / f"cxr_kaggle_{copied+1}{src.suffix.lower()}"
        idx = 1
        while out.exists():
            out = target_dir / f"cxr_kaggle_{copied+1}_{idx}{src.suffix.lower()}"
            idx += 1
        try:
            copy2(src, out)
            copied += 1
        except Exception:
            continue
    return copied


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--out-dir", default=str(Path("data") / "covid_ct_cxr" / "images"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    added = ensure_min_cxr_images(out_dir, args.min_count)
    print(f"Added {added} NIH CXR images to {out_dir}")


if __name__ == "__main__":
    main()



