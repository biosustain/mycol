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

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Matches cellpose.models.model_path(): cyto/cyto2/nuclei gain a "torch_0"
# suffix, everything else is used verbatim. These are the four base models
# offered in the Cellpose popover in src/panels/mask_editing_panel.py.
CELLPOSE_MODELS = [
    "cyto3",
    "cyto2torch_0",
    "cytotorch_0",
    "nucleitorch_0",
    # Size models are only a few KB and are used if a user ever switches to the
    # diameter-estimating Cellpose class, so they are cheap insurance.
    "size_cyto3.npy",
    "size_cyto2torch_0.npy",
    "size_cytotorch_0.npy",
    "size_nucleitorch_0.npy",
]
CELLPOSE_URL = "https://www.cellpose.org/models"

MOBILE_SAM_REPO = "dhkim2810/MobileSAM"
MOBILE_SAM_FILE = "mobile_sam.pt"


def _fetch_cellpose(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in CELLPOSE_MODELS:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  - {name}: already present")
            continue
        url = f"{CELLPOSE_URL}/{name}"
        print(f"  - {name}: downloading")
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except (urllib.error.URLError, TimeoutError) as exc:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"Failed to download {url}: {exc}") from exc
        tmp.replace(target)


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
