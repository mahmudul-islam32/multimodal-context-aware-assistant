"""Download public HAR datasets.

    python -m mcaa.data.download --dataset uci-har
    python -m mcaa.data.download --dataset wisdm

The SHA-256 of each archive is printed and written next to it so the exact
file used for a result can be reported. Pass ``--sha256`` to verify.
Licences: UCI HAR and WISDM are CC BY 4.0 (UCI ML Repository); cite them.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import httpx

SOURCES = {
    "uci-har": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "wisdm": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_nested(path: Path, dest: Path) -> None:
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)
    for inner in list(dest.rglob("*.zip")):
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
        inner.unlink()


def download(name: str, root: Path = Path("data/raw"), expected: str | None = None) -> Path:
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.zip"
    if not archive.exists():
        with httpx.stream("GET", SOURCES[name], follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(archive, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
    digest = sha256(archive)
    (dest / "SHA256").write_text(digest + "\n")
    if expected and digest != expected:
        raise RuntimeError(f"checksum mismatch for {name}: {digest} != {expected}")
    _extract_nested(archive, dest)
    print(f"{name}: {archive} sha256={digest}")
    return dest


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    p.add_argument("--root", default="data/raw")
    p.add_argument("--sha256")
    a = p.parse_args(argv)
    download(a.dataset, Path(a.root), a.sha256)


if __name__ == "__main__":
    main()
