"""Download the model weights Mycol fetches on first use into a bundle directory.

The app normally pulls MobileSAM from the Hugging Face Hub and the Cellpose
base models from cellpose.org the first time a user segments something. That
turns a "finished" install into a download that fails behind institutional
firewalls, at exactly the moment the user expects the app to work. Running this
at build time bakes both into the distributed folder instead.

The layout written here is the one src/bootstrap.py points MYCOL_MODELS_DIR and
CELLPOSE_LOCAL_MODELS_PATH at:

    <dest>/models/mobile_sam.pt   MobileSAM checkpoint
    <dest>/models/cellpose/       Cellpose base models

MobileSAM is written as a plain file rather than into the Hugging Face cache on
purpose: that cache links snapshots/ to blobs/ with symlinks, which do not
survive a zip round-trip on Windows.

Usage:
    python scripts/fetch_models.py <dest>
"""

import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Matches cellpose.models.model_path(): cyto/cyto2/nuclei gain a "torch_0"
# suffix, everything else is used verbatim. These are the four base models
# offered in the Cellpose popover in src/panels/mask_editing_panel.py.
# SHA-256 of the files shipped in v0.2.2, which came from cellpose.org.
CELLPOSE_MODELS = {
    "cyto3": "2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c",
    "cyto2torch_0": "0c2ff54f8cf38aeccceb119b72ae82285bbda2efb15cc493dde43de917d95d81",
    "cytotorch_0": "6a852487b98a3ad91e4e86c969cac520aaac13c609288aad8bd01d4cf76370c6",
    "nucleitorch_0": "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303",
    # Size models are only a few KB and are used if a user ever switches to the
    # diameter-estimating Cellpose class, so they are cheap insurance.
    "size_cyto3.npy": "87f91feb48019b4dfb0400e7bbb89aca5873368686815aa7b4431c0faee97fc5",
    "size_cyto2torch_0.npy": "c69a3688d61a93eb8da1592ff6f501d89a441b33668b1992170bd119557e0974",
    "size_cytotorch_0.npy": "87bcf523c13d72e61a99358babe531f67b3f0eca466eb4b6616d1430caeeebee",
    "size_nucleitorch_0.npy": "a79107c9f284b569dd65f3a7363014d613a3403241c062822f07b6f026b4d3ef",
}
# Tried in order. The mirror is a release on this repo holding the same files,
# so a cellpose.org outage (as on 2026-10-23) cannot block a release build.
CELLPOSE_URLS = [
    "https://github.com/biosustain/mycol/releases/download/models-v1",
    "https://www.cellpose.org/models",
]

MOBILE_SAM_REPO = "dhkim2810/MobileSAM"
MOBILE_SAM_FILE = "mobile_sam.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, target: Path, sha256: str) -> str | None:
    """Fetch url to target; return None on success or the reason it failed."""
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        return str(exc)
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        return f"checksum mismatch (got {got})"
    tmp.replace(target)
    return None


def _fetch_cellpose(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha256 in CELLPOSE_MODELS.items():
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  - {name}: already present")
            continue
        print(f"  - {name}: downloading")
        errors = []
        for base in CELLPOSE_URLS:
            url = f"{base}/{name}"
            reason = _download(url, target, sha256)
            if reason is None:
                break
            print(f"    {url}: {reason}")
            errors.append(f"{url}: {reason}")
        else:
            raise SystemExit(f"Failed to download {name}:\n  " + "\n  ".join(errors))


def _fetch_mobile_sam(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / MOBILE_SAM_FILE
    if target.exists() and target.stat().st_size > 0:
        print(f"  - {MOBILE_SAM_FILE}: already present")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed in the interpreter running this "
            "script. Run it with the bundle's python_main interpreter."
        )

    print(f"  - {MOBILE_SAM_FILE}: downloading")
    # local_dir writes a real file instead of a symlink into the shared cache.
    hf_hub_download(MOBILE_SAM_REPO, MOBILE_SAM_FILE, local_dir=str(dest))
    if not target.exists():
        raise SystemExit(f"Expected {target} after download, but it is missing.")
    # local_dir mode leaves a .cache/ of resume metadata that the bundle does not need.
    shutil.rmtree(dest / ".cache", ignore_errors=True)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    root = Path(sys.argv[1]).resolve()
    models = root / "models"

    print(f"Pre-baking model weights into {models}")
    print("[1/2] Cellpose base models...")
    _fetch_cellpose(models / "cellpose")
    print("[2/2] MobileSAM...")
    _fetch_mobile_sam(models)

    total = sum(f.stat().st_size for f in models.rglob("*") if f.is_file())
    print(f"Done. Bundled weights: {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
