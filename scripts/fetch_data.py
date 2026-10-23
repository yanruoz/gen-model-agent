"""Setup step, deliberately separate from an analysis run: downloads a dataset's
   data file if it is missing, from the `url:` declared in the '## Loading'
   section of that dataset's DATA_DESCRIPTION.md, and verifies `md5:` if one is
   declared. Any dataset can declare these, none is special-cased.

   The analysis pipeline itself never touches the network (CLAUDE.md rule 5),
   so a fresh clone runs this once before anything else:
       python scripts/fetch_data.py --dataset pbmc
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import loaders

ALLOWED_SCHEMES = ("http://", "https://", "file://")

# Some hosts (e.g. exampledata.scverse.org) return HTTP 403 for urllib's default
# "Python-urllib/x.y" User-Agent, so send an explicit one instead.
USER_AGENT = "gen-model-agent-fetch/1.0"


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(dataset):
    """Returns a short status string. Raises ValueError with a specific message
    on anything that would leave the dataset in an unverified state."""
    folder = loaders.DATA_ROOT / dataset
    if not folder.is_dir():
        raise ValueError(f"No folder {folder}; create it with a DATA_DESCRIPTION.md first.")
    spec = loaders.parse_loading_spec(folder / "DATA_DESCRIPTION.md")
    if "file" not in spec or "url" not in spec:
        raise ValueError(
            f"{dataset}: nothing to fetch. Declare both 'file:' and 'url:' (and ideally 'md5:') "
            "under '## Loading' in its DATA_DESCRIPTION.md."
        )
    if not spec["url"].startswith(ALLOWED_SCHEMES):
        raise ValueError(f"{dataset}: url must start with one of {ALLOWED_SCHEMES}, got {spec['url']!r}")

    target = folder / spec["file"]
    expected = spec.get("md5")

    if target.exists():
        if expected and md5_of(target) != expected:
            raise ValueError(
                f"{target} exists but its md5 ({md5_of(target)}) is not the declared {expected}. "
                "Not overwriting it; delete it yourself if it should be re-downloaded."
            )
        return f"already present: {target}" + (" (md5 verified)" if expected else " (no md5 declared)")

    partial = target.with_name(target.name + ".part")
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as resp, open(partial, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as e:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{dataset}: download from {spec['url']} failed: {e}") from e

    if expected and md5_of(partial) != expected:
        actual = md5_of(partial)
        partial.unlink(missing_ok=True)
        raise ValueError(f"{dataset}: downloaded file's md5 {actual} does not match the declared {expected}; discarded it.")
    partial.rename(target)
    return f"downloaded: {target}" + (" (md5 verified)" if expected else " (no md5 declared)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset's data file if it is missing.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    try:
        print(fetch(args.dataset))
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
