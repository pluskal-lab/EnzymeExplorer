"""Read ``drive/bundles.json`` and download / extract Drive artifacts.

The single source of truth for every setup script that needs to fetch a
Google-Drive-hosted zip or raw file. Handles both bundle types:

* ``type: "zip"`` — the URL points at a zip built by
  ``scripts/build_drive_bundles.py``. We download to the cache, verify
  its ``MANIFEST.txt`` (per-entry sha256), then apply the bundle's
  ``extract_recipes`` and run the ``sanity_checks``.
* ``type: "raw"`` — the URL points at a single file uploaded verbatim.
  We stream it straight to ``target_path`` and run the sanity checks.

Invocation:

    python -m scripts.drive_helper list [--required-by prod|dev|all]
    python -m scripts.drive_helper install-all --required-by prod|dev
    python -m scripts.drive_helper install <bundle_name>
    python -m scripts.drive_helper verify <bundle_name>

The env used to run this must have ``gdown`` on ``PATH`` — the setup
scripts arrange for that via ``conda run -n <env>``.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLES_JSON = REPO / "drive" / "bundles.json"
CACHE_DIR = REPO / ".drive_cache"


def _log(msg: str) -> None:
    print(f"[drive] {msg}", flush=True)


def _die(msg: str) -> None:
    print(f"[drive] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def load_bundles() -> dict:
    if not BUNDLES_JSON.exists():
        _die(f"{BUNDLES_JSON} not found")
    cfg = json.loads(BUNDLES_JSON.read_text())
    return cfg["bundles"]


def _expand(path: str) -> Path:
    """Expand ``$VAR`` in a path spec and return an absolute Path.

    Repo-relative paths (i.e. no leading env-var or ``/``) resolve
    against the repo root.
    """
    exp = os.path.expandvars(path)
    p = Path(exp)
    if not p.is_absolute():
        p = REPO / p
    return p


def _gdown(url: str, dst: Path) -> None:
    """Download ``url`` to ``dst`` atomically via gdown.

    Uses a ``.part`` sibling so a killed process never leaves a
    truncated file that the cache would treat as complete.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    subprocess.check_call(
        ["gdown", "--output", str(tmp), url]
    )
    os.replace(tmp, dst)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


MANIFEST_LINE_RE = re.compile(
    # <sha256>  <size>  <relpath>[  (extra-info)]
    #                              ^ trailing parenthesised comment (optional)
    # Legacy MANIFEST files (e.g. the pre-refactor reference-domains.zip)
    # record directory placeholders with a ``(n_files=N)`` sidecar; the
    # newer build_drive_bundles.py output never emits that.
    r"^([0-9a-f]{64})\s+(\d+)\s+(.+?)(?:\s+\([^)]*\))?\s*$"
)


def verify_manifest(zip_path: Path) -> None:
    """Verify per-entry sha256 in ``MANIFEST.txt`` against the archive.

    Robust to two MANIFEST dialects:

    * modern (produced by ``scripts/build_drive_bundles.py``) — pure
      ``<sha256>  <size>  <relpath>`` lines, one per file.
    * legacy — additionally records directory placeholders with a
      ``(n_files=N)`` trailing comment; entries ending in ``/`` are the
      directory summary, not a real archive entry.

    Directory summaries and lines that don't match the schema are
    skipped. Entries listed in the manifest but genuinely absent from
    the archive cause a hard failure (that's an actual upload bug).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            manifest_bytes = zf.read("MANIFEST.txt")
        except KeyError:
            _die(f"{zip_path.name}: MANIFEST.txt missing from zip")
        arcnames = set(zf.namelist())
        for raw in manifest_bytes.decode("utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = MANIFEST_LINE_RE.match(line)
            if not m:
                continue  # malformed; skip silently
            expected_sha, _size, relpath = m.group(1), m.group(2), m.group(3)
            if relpath.endswith("/"):
                continue  # directory placeholder — not a real entry
            if relpath not in arcnames:
                _die(
                    f"{zip_path.name}: manifest entry '{relpath}' missing "
                    f"from archive — zip is corrupt, delete {zip_path} "
                    f"and re-download"
                )
            with zf.open(relpath) as fh:
                h = hashlib.sha256()
                while True:
                    buf = fh.read(1 << 20)
                    if not buf:
                        break
                    h.update(buf)
            if h.hexdigest() != expected_sha:
                _die(
                    f"{zip_path.name}: entry '{relpath}' sha256 mismatch — "
                    f"delete {zip_path} and re-run to re-download"
                )


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def extract_zip(zip_path: Path, spec: dict) -> None:
    """Extract per ``spec['extract_recipes']``. Each recipe has:

    * ``patterns`` — list of shell globs matched against the arcname
      *after* ``strip_prefix`` has been removed.
    * ``target_dir`` — where to unpack (env-vars expanded; repo-relative
      if not absolute).
    * ``strip_prefix`` (optional) — a leading path component to remove
      from each arcname before it's matched against ``patterns`` and
      before it's written to ``target_dir``. Set this when the uploaded
      zip is wrapped in an extra outer folder (e.g. the legacy
      ``reference-domains.zip`` groups everything under
      ``reference-domains/``, so recipes use ``strip_prefix:
      "reference-domains/"`` and patterns like
      ``"martsDB_detected_domains/*"``).

    Extracts each match to ``target_dir / (arcname - strip_prefix)`` —
    exactly what ``unzip -q -o <zip> '<strip_prefix><pat>' -d <target>``
    followed by moving the stripped prefix up would produce.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n != "MANIFEST.txt"]
        for recipe in spec.get("extract_recipes", []):
            target = _expand(recipe["target_dir"])
            target.mkdir(parents=True, exist_ok=True)
            strip = recipe.get("strip_prefix", "")
            matched: list[tuple[str, str]] = []  # (arcname, stripped_relpath)
            for name in names:
                if name.endswith("/"):
                    continue  # directory placeholder
                if strip:
                    if not name.startswith(strip):
                        continue
                    rel = name[len(strip):]
                    if not rel:
                        continue
                else:
                    rel = name
                if _matches_any(rel, recipe["patterns"]):
                    matched.append((name, rel))
            if not matched:
                _log(
                    f"WARN: recipe patterns={recipe['patterns']} "
                    f"strip_prefix={strip!r} matched nothing in {zip_path.name}"
                )
                continue
            for name, rel in matched:
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                # Restore the Unix mode from the zip entry. ZipInfo stores
                # the mode in the upper 16 bits of external_attr for
                # entries created on Unix; if it's zero (e.g. the zip was
                # authored on a system that didn't record permissions) we
                # leave the default umask alone. This is what makes the
                # foldseek + USalign binaries in binaries.zip come out
                # executable instead of `Permission denied`.
                info = zf.getinfo(name)
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(dst, mode)


def check_sanity(spec: dict, label: str) -> None:
    """Every path in ``sanity_checks`` must exist post-install."""
    missing = []
    for check in spec.get("sanity_checks", []):
        p = _expand(check)
        if not p.exists():
            missing.append(str(p))
    if missing:
        _die(f"[{label}] sanity check failed — missing:\n  " + "\n  ".join(missing))


def install_bundle(name: str, spec: dict, *, force: bool = False) -> None:
    url = spec.get("url")
    if not url:
        _die(f"[{name}] url is null in bundles.json — upload the artifact and set url first")

    typ = spec.get("type", "zip")
    if typ == "raw":
        target = _expand(spec["target_path"])
        if target.exists() and not force:
            _log(f"[{name}] target already at {target} — skipping (--force to re-download)")
        else:
            _log(f"[{name}] downloading {url} → {target}")
            _gdown(url, target)
        check_sanity(spec, name)
        _log(f"[{name}] OK ({typ})")
        return

    # zip
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_zip = CACHE_DIR / spec["zip_name"]

    # Re-download if the outer sha256 doesn't match the recorded value.
    need_download = force or not cache_zip.exists()
    expected_sha = spec.get("sha256_zip")
    if cache_zip.exists() and not need_download and expected_sha:
        actual = _sha256_file(cache_zip)
        if actual != expected_sha:
            _log(f"[{name}] cached zip sha256 mismatch (have {actual[:12]}, want "
                 f"{expected_sha[:12]}) — re-downloading")
            need_download = True

    if need_download:
        _log(f"[{name}] downloading {url} → {cache_zip}")
        _gdown(url, cache_zip)
    else:
        _log(f"[{name}] reusing cached {cache_zip.name} ({cache_zip.stat().st_size:,} B)")

    _log(f"[{name}] verifying MANIFEST.txt …")
    verify_manifest(cache_zip)

    _log(f"[{name}] extracting per recipes …")
    extract_zip(cache_zip, spec)

    check_sanity(spec, name)
    _log(f"[{name}] OK (zip)")


def _fmt_status(spec: dict) -> str:
    url = "URL✓" if spec.get("url") else "URL·"
    typ = spec.get("type", "zip")
    return f"type={typ:<3}  {url}"


def cmd_list(args: argparse.Namespace, bundles: dict) -> None:
    w = max(len(n) for n in bundles)
    for name, spec in bundles.items():
        if args.required_by != "all" and args.required_by not in spec.get("required_by", []):
            continue
        print(f"  {name.ljust(w)}  {_fmt_status(spec)}  "
              f"required_by={spec.get('required_by')}")


def cmd_install_all(args: argparse.Namespace, bundles: dict) -> None:
    todo = [
        (n, s) for n, s in bundles.items()
        if args.required_by in s.get("required_by", [])
    ]
    _log(f"install-all --required-by {args.required_by}: {len(todo)} bundles")
    missing_urls = [n for n, s in todo if not s.get("url")]
    if missing_urls and not args.allow_missing_urls:
        _die(
            f"{len(missing_urls)} bundle(s) have no url yet — set them in "
            f"drive/bundles.json first, or pass --allow-missing-urls to skip: "
            + ", ".join(missing_urls)
        )
    for name, spec in todo:
        if not spec.get("url"):
            _log(f"[{name}] url missing — skipping")
            continue
        install_bundle(name, spec, force=args.force)
    _log("install-all done")


def cmd_install(args: argparse.Namespace, bundles: dict) -> None:
    if args.name not in bundles:
        _die(f"unknown bundle: {args.name}")
    install_bundle(args.name, bundles[args.name], force=args.force)


def cmd_verify(args: argparse.Namespace, bundles: dict) -> None:
    if args.name not in bundles:
        _die(f"unknown bundle: {args.name}")
    check_sanity(bundles[args.name], args.name)
    _log(f"[{args.name}] sanity OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="Show bundle URL / sanity status")
    sp.add_argument("--required-by", choices=["prod", "dev", "all"], default="all")

    sp = sub.add_parser("install-all", help="Install every bundle whose "
                        "required_by contains the given tag")
    sp.add_argument("--required-by", choices=["prod", "dev"], required=True)
    sp.add_argument("--force", action="store_true",
                    help="Re-download even if the cached zip's sha256 matches")
    sp.add_argument("--allow-missing-urls", action="store_true",
                    help="Warn+skip bundles whose url is null (default: fail loudly)")

    sp = sub.add_parser("install", help="Install a single named bundle")
    sp.add_argument("name")
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("verify", help="Run only the sanity_checks for a bundle")
    sp.add_argument("name")

    args = ap.parse_args()
    bundles = load_bundles()

    dispatch = {
        "list":        cmd_list,
        "install-all": cmd_install_all,
        "install":     cmd_install,
        "verify":      cmd_verify,
    }
    dispatch[args.cmd](args, bundles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
