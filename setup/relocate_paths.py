"""Rewrite stored absolute paths after moving data to a new server (or a new data folder).

chain/chain.json, the take .json files and colour records store absolute paths. After copying the data, replace the
old data-root prefix with the new one:

  $SG_PY setup/relocate_paths.py --old /share_2/users/umair_nawaz/World-Models/SANA-Work/PoC
  $SG_PY setup/relocate_paths.py --old /old/root --dry_run           # only report

Default --new is the current data root (SG_ROOT). Default files: every .json under chain/ and work/ (except the
queue). records/chain.json in the repository uses the placeholder "{ROOT}":

  $SG_PY setup/relocate_paths.py --old "{ROOT}" --files records/chain.json --out chain/chain.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import sg_paths  # noqa: E402


def swap(o, old, new, n):
    if isinstance(o, str):
        if old in o:   # a path value, or paths inside a command string
            n[0] += o.count(old)
            return o.replace(old, new)
        return o
    if isinstance(o, list):
        return [swap(v, old, new, n) for v in o]
    if isinstance(o, dict):
        return {k: swap(v, old, new, n) for k, v in o.items()}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="old data root (or the placeholder {ROOT})")
    ap.add_argument("--new", default=sg_paths.ROOT)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="with one --files entry: write here instead of in place")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    old = a.old.rstrip("/")
    new = a.new.rstrip("/") if a.new.startswith("{") else os.path.abspath(a.new).rstrip("/")
    files = a.files
    if files is None:
        files = [p for d in ("chain", "work") for p in glob.glob(os.path.join(new, d, "**", "*.json"), recursive=True)
                 if "/work/queue/" not in p]
    total = 0
    for p in files:
        try:
            d = json.load(open(p))
        except (OSError, ValueError) as e:
            print(f"skip {p}: {e}")
            continue
        n = [0]
        d2 = swap(d, old, new, n)
        total += n[0]
        if n[0] and not a.dry_run:
            out = a.out or p
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            json.dump(d2, open(out + ".tmp", "w"), indent=1)
            os.replace(out + ".tmp", out)
        if n[0] or a.out:
            print(f"{n[0]:5d} paths  {p}" + (f" -> {a.out}" if a.out else ""))
    print(f"{'would rewrite' if a.dry_run else 'rewrote'} {total} paths in {len(files)} files: {old} -> {new}")


if __name__ == "__main__":
    main()
