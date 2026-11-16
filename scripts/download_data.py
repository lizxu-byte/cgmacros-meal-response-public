"""Download CGMacros v1.0.0 from PhysioNet (open access, CC BY-NC-SA 4.0) into data/.

The archive is checked against the SHA256 published by PhysioNet. Only the CSV
files are extracted; the meal photographs stay inside the zip and are never
extracted or committed. Usage:  python scripts/download_data.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://physionet.org/files/cgmacros/1.0.0/"
ZIP_NAME = "CGMacros_dateshifted365.zip"
ZIP_SHA256 = "05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d"
EXTRA = ["DataDictionary_Bio.csv", "DataDictionary_CGMacros-00X.csv", "SHA256SUMS.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    data = Path(args.data_dir)
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / ZIP_NAME
    if not zpath.exists() or sha256(zpath) != ZIP_SHA256:
        fetch(BASE + ZIP_NAME, zpath)
    digest = sha256(zpath)
    if digest != ZIP_SHA256:
        print(f"SHA256 mismatch: {digest}", file=sys.stderr)
        return 1
    print("SHA256 ok")
    for name in EXTRA:
        if not (raw / name).exists():
            fetch(BASE + name, raw / name)
    with zipfile.ZipFile(zpath) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        for n in csvs:
            target = (data / n).resolve()
            if data.resolve() not in target.parents:  # guard against path traversal
                raise RuntimeError(f"unsafe path in archive: {n}")
            z.extract(n, data)
        n_photos = sum(n.lower().endswith((".jpg", ".jpeg", ".png")) for n in z.namelist())
    print(f"extracted {len(csvs)} CSV files to {data / 'CGMacros'}; {n_photos} photo files left in the zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
