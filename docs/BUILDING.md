# Building the Windows bundle

The bundle is a self-contained folder: it carries its own Python interpreters,
all dependencies, and the model weights, so the target machine needs no Python,
pip, uv or git.

## Prerequisites

Building (not running) needs three things on the build machine:

- **Windows** — the bundle embeds Windows Python builds, so it must be built on Windows.
- **PowerShell 7+** — `winget install Microsoft.PowerShell`
- **uv** — see the [install instructions](https://docs.astral.sh/uv/getting-started/installation/)
- **Rust** — the native launcher is compiled with cargo, via [rustup](https://rustup.rs)

## Build

```powershell
./scripts/make_dist.ps1 -Version 0.2.0                  # CPU bundle (default)
./scripts/make_dist.ps1 -Version 0.2.0 -Variant cuda    # NVIDIA CUDA 12.6 bundle
./scripts/make_dist.ps1 -Version 0.2.0 -Variant both    # both, sequentially
```

Output lands in `dist\`:

```
dist\mycol-windows-cpu\              the bundle folder
dist\mycol-windows-cpu-v0.2.0.zip    the release asset
```

Pass `-SkipModels` to skip pre-baking the weights. That makes local test builds
much faster, but the resulting bundle downloads models on first use — which is
the behaviour releases exist to avoid. Never ship a `-SkipModels` build.

## Verify

```powershell
./scripts/verify_dist.ps1 -DistDir dist/mycol-windows-cpu
```

This checks the required files are present, that `config.toml` parses, that the
heavy imports work in both embedded interpreters, and that the app actually
serves a page. CI runs it after every build; run it locally before publishing a
bundle by hand.

## Testing a branch before merging

The PowerShell build can only run on Windows, so CI is the place to exercise it.

**Every push to `main`, `windows_local` or `release/**` builds the CPU bundle**,
runs `verify_dist.ps1` against it, and uploads the zip as a run artifact. Nothing
is published. Open the run from the **Actions** tab and download the artifact to
try the bundle on a real Windows machine.

Doc-only changes (`docs/`, `manuscript/`, `*.md`, the case studies and notebooks)
are excluded, so they do not spend half an hour of Windows runner time.

To exercise the full release path — both variants, plus the release upload —
without touching `main`, push a pre-release tag from the branch:

```bash
git tag v0.2.0-rc1 && git push origin v0.2.0-rc1
```

Tag triggers are not branch-scoped, so this works from any branch. The tag's
hyphen marks the GitHub release as a pre-release. It is still created as a draft,
so nothing is public until you publish it. Delete the tag and the draft when done:

```bash
git push --delete origin v0.2.0-rc1 && git tag -d v0.2.0-rc1
```

> [!NOTE]
> The **Run workflow** button (`workflow_dispatch`) only appears once this
> workflow file is on the repository's *default* branch. Until then, use a branch
> push or a pre-release tag.

## Release

Once the branch is merged, pushing a `v*` tag runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml), which builds
both variants on separate runners, verifies them, and attaches the zips to a
**draft** release.

```bash
git tag v0.2.0 && git push origin v0.2.0
```

Then review the draft on the Releases page and click **Publish release**. Until
you do, `releases/latest` shows nothing — drafts are invisible to everyone else.

> [!IMPORTANT]
> The build needs **Settings → Actions → General → Workflow permissions** set to
> *Read and write*, or the release cannot be created.

> [!WARNING]
> GitHub caps a single release asset at 2 GiB. The CUDA bundle can approach that;
> `make_dist.ps1` warns when a zip exceeds it. If that happens, host the CUDA
> build elsewhere rather than letting the upload fail during the release.

## What the script does

1. Downloads the Python **embeddable** distributions — 3.13 for the app, 3.10 for
   the training worker — into `bin\python_main` and `bin\python_worker`, and
   re-enables `site-packages` in each `._pth`.
2. Exports pinned requirements from `uv.lock` and `src/training/uv.lock`, then
   installs them into each interpreter with `uv pip install --python`.
   For `-Variant cuda`, CUDA torch is then installed over the top.
3. Copies `src\`, `app.py`, `bootstrap.py`, `logo.png`, `LICENSE` and
   `.streamlit\` into the bundle, and writes a short `README.txt` for users.
4. Pre-bakes the model weights via
   [`scripts/fetch_models.py`](../scripts/fetch_models.py) so first use needs no
   network. `bootstrap.py` points `CELLPOSE_LOCAL_MODELS_PATH` and
   `MYCOL_MODELS_DIR` at them.
5. Compiles the Rust launcher into `mycol.exe` (windowed) and `mycol_debug.exe`
   (keeps a console, for diagnosing failures).
6. Zips the folder.

## Why is it so large?

Each bundle carries two complete Python interpreters, PyTorch, and ~150 MB of
model weights. The CUDA variant adds the NVIDIA runtime libraries on top, which
is most of the difference between the ~1 GB and ~3 GB downloads.

## Torch: CPU by default

`pyproject.toml` points torch at the **CPU** index for Windows and Linux. The
CUDA wheels pull in roughly 2.5 GB of `nvidia-*` packages that most laptops
cannot use, and that download was the most common way `uv sync` failed partway
through. macOS resolves from PyPI, whose wheels already carry MPS support.

CUDA is opt-in, either through `-Variant cuda` here or, in a source checkout:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## macOS and Linux

`scripts/make_dist.sh` is **not** release-ready — see the warning at the top of
that file. Building distributable macOS artifacts needs a relocatable interpreter,
a real `.app` bundle, and Apple notarization, none of which that script does yet.
