# Lightweight fetcher for COVID-19-CT-CXR repo images via GitHub API
# Downloads up to N images into data/covid_ct_cxr/images/

import os
import json
import sys
import time
import urllib.request
import urllib.error
from typing import List, Dict

ROOT_API = "https://api.github.com/repos/ncbi-nlp/COVID-19-CT-CXR/contents"
USER_AGENT = "he-multimodal-agent/0.1"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def http_get_json(url: str) -> List[Dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)


def http_download(url: str, out_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(resp.read())


def is_image_name(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in IMG_EXTS)


def collect_image_urls(max_files: int = 30) -> List[Dict]:
    """Traverse repo contents to collect image file entries (download_url + path)."""
    queue = [""]
    results = []
    seen_dirs = set()

    while queue and len(results) < max_files:
        rel = queue.pop(0)
        url = ROOT_API + ("/" + rel if rel else "")
        try:
            entries = http_get_json(url)
        except urllib.error.HTTPError as e:
            # Rate limit or not found; backoff and continue
            print(f"WARN: HTTP {e.code} for {url}; sleep and continue")
            time.sleep(1.0)
            continue
        except Exception as e:
            print(f"ERROR listing {url}: {e}")
            continue

        # entries: list of dicts with type ('file' or 'dir'), name, path, download_url
        for ent in entries:
            typ = ent.get("type")
            name = ent.get("name", "")
            path = ent.get("path", "")
            if typ == "dir":
                if path not in seen_dirs:
                    seen_dirs.add(path)
                    queue.append(path)
            elif typ == "file" and is_image_name(name):
                if ent.get("download_url"):
                    results.append({"download_url": ent["download_url"], "path": path})
                    if len(results) >= max_files:
                        break
        time.sleep(0.2)  # polite pacing
    return results


def main():
    max_files = 30
    if len(sys.argv) >= 2:
        try:
            max_files = int(sys.argv[1])
        except ValueError:
            pass

    out_dir = os.path.join("data", "covid_ct_cxr", "images")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Listing repo to collect up to {max_files} images...")
    files = collect_image_urls(max_files=max_files)
    print(f"Found {len(files)} candidate images.")

    downloaded = 0
    for i, ent in enumerate(files, 1):
        url = ent["download_url"]
        fname = os.path.basename(ent["path"]) or f"img_{i}.jpg"
        # Ensure unique filenames
        out_path = os.path.join(out_dir, fname)
        base, ext = os.path.splitext(out_path)
        if os.path.exists(out_path):
            out_path = f"{base}_{i}{ext}"
        try:
            http_download(url, out_path)
            downloaded += 1
            if downloaded % 5 == 0:
                print(f"Downloaded {downloaded}/{len(files)}")
        except Exception as e:
            print(f"ERROR downloading {url}: {e}")
    print(f"Done. Downloaded {downloaded} images to {out_dir}")


if __name__ == "__main__":
    main()