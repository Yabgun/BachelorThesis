import os, json, random, zipfile, io, re
import argparse
from pathlib import Path
import requests

OWNER = "ncbi-nlp"; REPO = "COVID-19-CT-CXR"
HEADERS = {"User-Agent": "he-poc-fetcher"}
ROOT = Path("data/covid_ct_cxr"); IMG_DIR = ROOT/"images"; REL_DIR = ROOT/"releases"
IMG_DIR.mkdir(parents=True, exist_ok=True); REL_DIR.mkdir(parents=True, exist_ok=True)
IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

T_CONN, T_READ = 10, 15

def list_releases():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"
    r = requests.get(url, headers=HEADERS, timeout=(T_CONN,T_READ))
    r.raise_for_status(); return r.json()

def pick_assets(release):
    assets = release.get("assets", [])
    zip_assets = [a for a in assets if a.get("name","" ).endswith("subfigures_cxr_ct_gold.zip")]
    json_assets = [a for a in assets if a.get("name","" ).endswith(".litcovid.released.image.json")]
    return zip_assets, json_assets

def download_asset(asset):
    url = asset["browser_download_url"]; name = asset["name"]; dest = REL_DIR/name
    if dest.exists(): return dest
    print(f"Downloading asset {name}...")
    with requests.get(url, headers=HEADERS, stream=True, timeout=(T_CONN,T_READ)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk: f.write(chunk)
    return dest

def figure_urls_from_json(json_path):
    print(f"Parsing {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    urls = []
    for row in (data if isinstance(data, list) else data.get("rows", [])):
        for key in ("figure_url","figureUrl","url","image_url"):
            u = row.get(key)
            if u: urls.append(u); break
    print(f"Found {len(urls)} figure URLs")
    return list(dict.fromkeys(urls))

def save_image(url, out_dir:Path, idx:int, pmcid:str=""):
    try:
        r = requests.get(url, headers=HEADERS, timeout=(T_CONN,T_READ), allow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type","")
        ext = ".jpg" if "jpeg" in ct else ".png" if "png" in ct else None
        if not ext:
            m = re.search(r"\.(png|jpg|jpeg|tif|tiff)(?:\?|$)", url, re.I)
            ext = f".{m.group(1).lower()}" if m else ".jpg"
        fname = f"{pmcid or 'img'}_{idx}{ext}"
        (out_dir/fname).write_bytes(r.content)
        return True
    except Exception as e:
        print(f"Skip {url}: {e}"); return False

def extract_from_zip(zip_path:Path, max_images:int):
    n = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXTS]
        random.shuffle(names)
        for i, name in enumerate(names):
            if n >= max_images: break
            try:
                data = z.read(name)
                hint = "ct" if "ct" in name.lower() else "cxr" if "cxr" in name.lower() else "img"
                out = IMG_DIR/f"{hint}_{i}{Path(name).suffix.lower()}"
                out.write_bytes(data); n += 1
            except Exception as e:
                print(f"Skip in zip {name}: {e}")
    return n

def fetch_images(max_images:int=30):
    releases = list_releases()
    if not releases: raise RuntimeError("No releases found for COVID-19-CT-CXR")
    zip_assets, json_assets = pick_assets(releases[0])
    print(f"Assets: zip={len(zip_assets)}, json={len(json_assets)}")
    total = 0
    if zip_assets:
        for a in zip_assets:
            zp = download_asset(a)
            print(f"Extracting up to {max_images-total} images from {zp.name}...")
            total += extract_from_zip(zp, max_images - total)
            if total >= max_images: break
    elif json_assets:
        urls = []
        for a in json_assets:
            jp = download_asset(a)
            urls.extend(figure_urls_from_json(jp))
        random.shuffle(urls)
        print(f"Downloading up to {max_images} images from figure URLs...")
        for i, u in enumerate(urls):
            if total >= max_images: break
            ok = save_image(u, IMG_DIR, i)
            total += 1 if ok else 0
    else:
        raise RuntimeError("No suitable assets found (JSON or subfigure zip)")
    return total

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-images", type=int, default=int(os.environ.get("MAX_IMAGES", "30")))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = fetch_images(max_images=args.max_images)
    print(f"Fetched {count} images into {IMG_DIR}")