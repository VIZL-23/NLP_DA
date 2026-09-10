"""Resumable download of the merged crack-segmentation dataset (image + mask).

Written resumable on purpose: this machine's connection drops mid-transfer
(observed repeatedly - the CLIP fetch stalled at 0 bytes, and a plain
urlretrieve of these files died at 38 MB of 159 MB). A restart-from-zero
downloader is unusable here, so this one uses HTTP Range to continue from
whatever is already on disk, retries each file, and verifies the final size
against Content-Length before declaring success.

Run:
    python scripts/download_cracks.py
Re-run it after any failure; completed files are skipped and partial ones
resume.
"""

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "datasets" / "CRACK500" / "parquet"

BASE = ("https://huggingface.co/datasets/zskitecho/crack-segmentation-dataset"
        "/resolve/main/data/")
FILES = [
    "test-00000-of-00001-3da1883d5c5df420.parquet",
    "train-00000-of-00003-06e580b36935d137.parquet",
    "train-00001-of-00003-b7d4634b852df18c.parquet",
    "train-00002-of-00003-d7c3d9a5fc0295d1.parquet",
]

CHUNK = 1 << 16
MAX_ATTEMPTS = 40


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        # HF serves LFS objects via redirect; X-Linked-Size carries the real
        # length when Content-Length describes the pointer instead.
        return int(r.headers.get("X-Linked-Size") or r.headers["Content-Length"])


def fetch(name: str) -> bool:
    url, dst = BASE + name, OUT / name
    total = remote_size(url)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= total:
            print(f"  complete: {name}  ({have / 1e6:.0f} MB)")
            return True

        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "ab") as f:
                t0, start = time.time(), have
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
            rate = (have - start) / max(time.time() - t0, 1e-6) / 1e3
            print(f"  attempt {attempt}: {have / 1e6:.0f}/{total / 1e6:.0f} MB "
                  f"({rate:.0f} kB/s)", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"  attempt {attempt}: dropped at {have / 1e6:.0f} MB "
                  f"({type(e).__name__}) - resuming", flush=True)
            time.sleep(min(5 * attempt, 30))

    final = dst.stat().st_size if dst.exists() else 0
    print(f"  GAVE UP on {name}: {final / 1e6:.0f}/{total / 1e6:.0f} MB")
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in FILES:
        print(f"{name}")
        ok &= fetch(name)
    print("\nALL COMPLETE" if ok else "\nINCOMPLETE - re-run to resume")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
