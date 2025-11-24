import os
import re
from pathlib import Path
from typing import List, Dict
from scripts.fetch_covid_ct_cxr import collect_image_urls, http_download


def sanitize_ext(name: str) -> str:
    lower = name.lower()
    m = re.search(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", lower)
    return m.group(0) if m else ".jpg"


def main(max_files: int = 100) -> None:
    out_dir = Path("data") / "covid_ct_cxr" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Dict] = collect_image_urls(max_files=max_files)
    downloaded = 0
    for i, ent in enumerate(files, 1):
        url = ent["download_url"]
        ext = sanitize_ext(url)
        base = f"cxr_aug_{i}"
        out_path = out_dir / f"{base}{ext}"
        idx = 1
        while out_path.exists():
            out_path = out_dir / f"{base}_{idx}{ext}"
            idx += 1
        try:
            http_download(url, str(out_path))
            downloaded += 1
        except Exception:
            continue
    print(f"Downloaded {downloaded} augmented CXR images into {out_dir}")


if __name__ == "__main__":
    main()



