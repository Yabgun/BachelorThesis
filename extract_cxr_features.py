import csv
from pathlib import Path
import numpy as np
from PIL import Image

IMG_DIR = Path('data/covid_ct_cxr/images')
OUT_PATH = Path('data/covid_ct_cxr/cxr_features.csv')

# Process only files whose name starts with 'cxr_' by default
MODALITY_PREFIX = 'cxr_'


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')  # grayscale
    arr = np.asarray(img, dtype=np.float32)
    return arr


def entropy(arr: np.ndarray) -> float:
    # Shannon entropy over 256-level histogram
    hist = np.bincount(arr.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(arr: np.ndarray, thresh: float = 20.0) -> float:
    # Use simple gradients via np.gradient as a Sobel-like approximation
    gx, gy = np.gradient(arr)
    mag = np.sqrt(gx * gx + gy * gy)
    edges = (mag > thresh).sum()
    return float(edges) / float(arr.size)


def main():
    rows = []
    for p in sorted(IMG_DIR.glob('*.*')):
        name = p.name.lower()
        if not name.startswith(MODALITY_PREFIX):
            continue
        arr = load_gray(p)
        mean_int = float(arr.mean())
        std_int = float(arr.std())
        ed = edge_density(arr)
        ent = entropy(arr)
        rows.append({
            'image_path': str(p),
            'width': arr.shape[1],
            'height': arr.shape[0],
            'cxr_mean_intensity': mean_int,
            'cxr_std_intensity': std_int,
            'cxr_edge_density': ed,
            'cxr_entropy': ent,
        })

    if not rows:
        raise RuntimeError('No CXR images found with prefix cxr_ in data/covid_ct_cxr/images')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'image_path','width','height','cxr_mean_intensity','cxr_std_intensity','cxr_edge_density','cxr_entropy'
        ])
        w.writeheader()
        w.writerows(rows)

    print(f'Wrote {len(rows)} rows to {OUT_PATH}')


if __name__ == '__main__':
    main()