#!/usr/bin/env python3
"""Fetch / copy hash-locked ONNX assets. Standard library only; not a runtime dependency."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-source", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "release/ppocrv6-onnx-manifest.json").read_text("utf-8"))
    for asset in manifest["assets"]:
        destination = args.output / asset["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != asset["sha256"]:
                raise ValueError(f"existing asset hash mismatch: {destination}; not overwriting")
            print(f"verified {asset['path']}", flush=True)
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        if args.local_source:
            source = args.local_source / asset["path"]
            if digest(source) != asset["sha256"]:
                raise ValueError(f"source asset hash mismatch: {source}")
            shutil.copyfile(source, temporary)
        else:
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "lw.PPOCR.Java-model-fetch"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                total = 0
                for block in iter(lambda: response.read(1 << 20), b""):
                    total += len(block)
                    if total > 256 * 1024 * 1024:
                        raise ValueError("download exceeds model size limit")
                    out.write(block)
        if digest(temporary) != asset["sha256"]:
            raise ValueError(f"downloaded asset hash mismatch: {asset['path']}")
        temporary.rename(destination)
        print(f"verified {asset['path']}", flush=True)
    print("ONNX assets verified (7 models, 2 dictionaries)")

if __name__ == "__main__":
    main()
