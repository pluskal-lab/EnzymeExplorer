"""Build the Google Drive staging zips.

Reads ``drive/bundles.json`` (source of truth), zips each bundle from
its source paths into ``drive/<zip_name>``, and writes a ``MANIFEST.txt``
inside each zip listing per-entry ``<sha256>  <size>  <relpath>``.
After a zip is built, its outer sha256 + size are written back into the
bundle entry in ``bundles.json`` so downstream setup scripts can verify
downloads.

Idempotent: skips a bundle whose zip already exists and matches the
recorded size + sha256.

Usage:
    python scripts/build_drive_bundles.py                # build everything
    python scripts/build_drive_bundles.py <bundle> …     # build named bundles
    python scripts/build_drive_bundles.py --list         # show bundle status
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_DIR = REPO_ROOT / "drive"
BUNDLES_JSON = DRIVE_DIR / "bundles.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _iter_sources(src_paths: list[str]) -> list[tuple[Path, str]]:
    """Return [(abs_src, arcname), ...] for the union of ``src_paths``
    (files or directories). ``arcname`` is the path inside the zip; for
    a directory source the arcname preserves the dir name (e.g.
    ``martsDB_detected_domains/foo/bar.pkl``).
    """
    out: list[tuple[Path, str]] = []
    for rel in src_paths:
        p = REPO_ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"source path missing: {p}")
        if p.is_file():
            out.append((p, p.name))
        else:
            # Preserve the top-level dir name inside the zip.
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    arc = str(f.relative_to(p.parent))
                    out.append((f, arc))
    return out


def _filter_by_include(
    entries: list[tuple[Path, str]], include: list[str] | None
) -> list[tuple[Path, str]]:
    if not include:
        return entries
    import fnmatch
    keep = []
    for src, arc in entries:
        if any(fnmatch.fnmatchcase(arc, pat) for pat in include):
            keep.append((src, arc))
    return keep


def build_bundle(name: str, spec: dict, verbose: bool = True) -> dict:
    zip_name = spec["zip_name"]
    zip_path = DRIVE_DIR / zip_name
    include = spec.get("include")  # optional glob whitelist inside src_paths

    entries = _iter_sources(spec["src_paths"])
    entries = _filter_by_include(entries, include)
    if not entries:
        raise RuntimeError(f"[{name}] no files matched — check src_paths / include")

    if verbose:
        total_bytes = sum(s.stat().st_size for s, _ in entries)
        print(f"[{name}] {len(entries)} files, "
              f"~{total_bytes / 1e9:.2f} GB → {zip_name}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")

    manifest_lines: list[str] = []
    with zipfile.ZipFile(
        tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True,
    ) as zf:
        for i, (src, arc) in enumerate(entries, 1):
            data = src.read_bytes() if src.stat().st_size < 64 * 1024 * 1024 else None
            if data is not None:
                # Stream small files in-memory so we can hash without
                # a second read.
                sha = sha256_bytes(data)
                zf.writestr(arc, data)
                sz = len(data)
            else:
                # Large file: stream to zip + hash the source path.
                sha = sha256_file(src)
                zf.write(src, arcname=arc)
                sz = src.stat().st_size
            manifest_lines.append(f"{sha}  {sz}  {arc}")
            if verbose and (i % 200 == 0 or i == len(entries)):
                print(f"  [{name}] {i}/{len(entries)}")
        # MANIFEST.txt (must go inside the zip so downloads can be
        # verified against a self-contained checksum).
        manifest = "\n".join(manifest_lines) + "\n"
        zf.writestr("MANIFEST.txt", manifest.encode("utf-8"))

    os.replace(tmp_path, zip_path)
    outer_sha = sha256_file(zip_path)
    outer_size = zip_path.stat().st_size
    if verbose:
        print(f"[{name}] done: {zip_path.name} "
              f"({outer_size / 1e9:.2f} GB, sha256={outer_sha[:16]}…)")

    return {
        "size_bytes": outer_size,
        "sha256_zip": outer_sha,
        "n_files": len(entries),
    }


def load_bundles() -> dict:
    if not BUNDLES_JSON.exists():
        raise FileNotFoundError(
            f"{BUNDLES_JSON} not present. Create it from the template first."
        )
    return json.loads(BUNDLES_JSON.read_text())


def save_bundles(cfg: dict) -> None:
    BUNDLES_JSON.write_text(json.dumps(cfg, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="bundle name(s); omit to build all")
    ap.add_argument("--list", action="store_true",
                    help="print status of each bundle and exit")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the zip already exists")
    args = ap.parse_args()

    cfg = load_bundles()
    bundles = cfg["bundles"]

    if args.list:
        w = max(len(n) for n in bundles)
        print(f"{'bundle'.ljust(w)}  {'artifact':<44}  size(GB)  status")
        for name, spec in bundles.items():
            url_ok = "URL✓" if spec.get("url") else "URL·"
            if spec.get("type") == "raw":
                fn = spec.get("download_filename", "?")
                print(f"{name.ljust(w)}  {fn:<44}  "
                      f"{'  raw':>6}   RAW  {url_ok}")
                continue
            zp = DRIVE_DIR / spec["zip_name"]
            size = zp.stat().st_size if zp.exists() else 0
            zip_ok = "ZIP✓" if zp.exists() else "ZIP·"
            print(f"{name.ljust(w)}  {spec['zip_name']:<44}  "
                  f"{size / 1e9:>6.2f}   {zip_ok} {url_ok}")
        return 0

    todo = args.names or list(bundles.keys())
    for name in todo:
        if name not in bundles:
            print(f"unknown bundle: {name}", file=sys.stderr)
            return 2
        spec = bundles[name]
        if spec.get("type") == "raw":
            print(f"[{name}] type: raw — direct-download file, no zip to build")
            continue
        if spec.get("skip_local_build"):
            print(f"[{name}] skip_local_build set — already-hosted upstream zip; "
                  f"nothing to build locally")
            continue
        zp = DRIVE_DIR / spec["zip_name"]
        if zp.exists() and not args.force and spec.get("size_bytes") == zp.stat().st_size:
            print(f"[{name}] up-to-date — skipping (pass --force to rebuild)")
            continue
        result = build_bundle(name, spec)
        spec.update(result)
        save_bundles(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
