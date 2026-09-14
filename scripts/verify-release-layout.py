#!/usr/bin/env python3
"""Verify the public lw.PPOCR.Java Release directory contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_FILES = {
    "det.lwm",
    "cls.lwm",
    "rec.lwm",
    "ppocr_keys.txt",
    "sample.jpg",
}

ROOT_FILES = {
    "QUICKSTART.md",
    "README.md",
    "README.zh-CN.md",
    "CHANGELOG.md",
    "LICENSE",
    "THIRD-PARTY-NOTICES.md",
}

DOCUMENTS_WITH_CANONICAL_MODEL_PATH = {
    "QUICKSTART.md",
    "README.md",
    "README.zh-CN.md",
    "docs/installation.md",
    "docs/models.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty release file: {relative}")
    return path


def require_single_jar(root: Path, artifact: str) -> None:
    matches = list((root / "lib").glob(f"{artifact}-*.jar"))
    if len(matches) != 1 or matches[0].stat().st_size == 0:
        names = ", ".join(path.name for path in matches) or "none"
        raise SystemExit(f"expected one {artifact} JAR, found: {names}")


def verify(root: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"release root is not a directory: {root}")

    for relative in sorted(ROOT_FILES):
        require_file(root, relative)
    for directory in ("docs", "licenses", "lib", "models/ppocrv6-tiny"):
        if not (root / directory).is_dir():
            raise SystemExit(f"missing release directory: {directory}")

    for artifact in ("lw-ppocr-core", "lw-ppocr-imageio", "lw-ppocr-vector"):
        require_single_jar(root, artifact)

    model_root = root / "models" / "ppocrv6-tiny"
    manifest_path = require_file(root, "models/ppocrv6-tiny/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or set(manifest_files) != MODEL_FILES:
        raise SystemExit("release model manifest file set does not match the layout contract")

    for name in sorted(MODEL_FILES):
        model = require_file(root, f"models/ppocrv6-tiny/{name}")
        expected = manifest_files[name]
        actual = sha256(model)
        if actual != expected:
            raise SystemExit(f"{name}: expected SHA-256 {expected}, got {actual}")

    for old_path in ("models/det.lwm", "models/cls.lwm", "models/rec.lwm",
                     "models/ppocr_keys.txt"):
        if (root / old_path).exists():
            raise SystemExit(f"obsolete flat model path is present: {old_path}")

    canonical = "models/ppocrv6-tiny"
    for relative in sorted(DOCUMENTS_WITH_CANONICAL_MODEL_PATH):
        document = require_file(root, relative)
        if canonical not in document.read_text(encoding="utf-8"):
            raise SystemExit(f"{relative} does not reference the canonical model path")

    print(f"release layout: verified ({root})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="staged, uncompressed Release root")
    args = parser.parse_args()
    verify(args.root)


if __name__ == "__main__":
    main()
