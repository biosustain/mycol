# Building the desktop bundles

Each bundle is self-contained: it carries its own Python interpreters, all
dependencies, and the model weights, so the target machine needs no Python, pip,
uv or git.

A bundle can only be built on the platform — and on macOS the architecture — it
targets, because it embeds a real interpreter and architecture-specific wheels.
Windows is covered first; [macOS](#macos) is further down.

# Windows

## Prerequisites

Building (not running) needs four things on the build machine:

- **Windows** — the bundle embeds Windows Python builds, so it must be built on Windows.
- **PowerShell 7+** — `winget install Microsoft.PowerShell`
- **uv** — see the [install instructions](https://docs.astral.sh/uv/getting-started/installation/)
- **Rust** — the native launcher is compiled with cargo, via [rustup](https://rustup.rs)

## Build

```powershell
./scripts/windows/make_dist.ps1 -Version 0.2.0                  # CPU bundle (default)
./scripts/windows/make_dist.ps1 -Version 0.2.0 -Variant cuda    # NVIDIA CUDA 12.6 bundle
./scripts/windows/make_dist.ps1 -Version 0.2.0 -Variant both    # both, sequentially
```

Output lands in `dist\`:

```
dist\mycol-windows-cpu\              the bundle folder
dist\mycol-windows-cpu.zip          the release asset
```

Pass `-SkipModels` to skip pre-baking the weights. That makes local test builds
much faster, but the resulting bundle downloads models on first use — which is
the behaviour releases exist to avoid. Never ship a `-SkipModels` build.

## Check before you build

```bash
python scripts/windows/check_wheels.py
```

Runs in seconds on any platform and needs no Windows machine. It evaluates the
exported requirements against the bundle's target environment (Windows AMD64,
CPython 3.12 and 3.10) and fails if any package would be built from source
instead of installed from a wheel.

Source builds are what break the Windows build, because the embeddable
distribution ships no headers for a compiler to find. Two shipped in a row:

- `torchvision` resolving to `+d801a34`, a local build with only `win_arm64` wheels
- `numpy` 2.0.2 — capped by cellpose's `numpy<2.1` — having no cp313 wheel

The second is why the app bundle pins **Python 3.12, not 3.13**. CI runs this
check on Linux before it starts a Windows runner.

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

> [!IMPORTANT]
> Tagged releases build the **CPU bundle only**. A working CUDA bundle is ~5.7 GB
> and GitHub caps a release asset at 2 GiB, so attaching it always fails. Build
> one on demand with `workflow_dispatch` (variant `cuda`) and host it elsewhere,
> or point GPU users at a source install:
> `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

> [!WARNING]
> GitHub caps a single release asset at 2 GiB. The CUDA bundle can approach that;
> `make_dist.ps1` warns when a zip exceeds it. If that happens, host the CUDA
> build elsewhere rather than letting the upload fail during the release.

## What the script does

1. Downloads the Python **embeddable** distributions — 3.12 for the app, 3.10 for
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

# macOS

`scripts/macos/make_dist.sh` builds `Mycol.app` and a `.dmg`. It needs **uv** and
**Rust** on the build machine, and must run on the architecture it targets —
there is no cross-compiling, because the Python wheels are architecture-specific.

```bash
./scripts/macos/make_dist.sh --version 0.2.0        # builds for this Mac's architecture
./scripts/macos/verify_dist.sh dist/Mycol.app
```

Output:

```
dist/Mycol.app
dist/mycol-macos-arm64.dmg            (~1 GB; the .app is ~2.2 GB unpacked)
```

### Fast iteration: the dev app

A full `make_dist.sh` run takes minutes, which is far too slow to sit in an
edit-test loop. `scripts/macos/dev_app.sh` builds a bundle that **symlinks** `src/`,
`app.py` and `bootstrap.py` back to the working tree, and keeps the interpreters
and model weights in a cache under `build/dev/`:

```bash
./scripts/macos/dev_app.sh              # build (or refresh) and launch
./scripts/macos/dev_app.sh --no-open    # build only
./scripts/macos/dev_app.sh --reset      # discard the cached environment and rebuild
```

- **First run** adopts the environment from an existing `dist/Mycol.app` if there
  is one. The copy uses `cp -c`, which clones on APFS — measured at 0 MB of real
  disk for a 140 MB tree, so the cache is effectively free.
- **Later runs** take about 0.1 s.
- **Code edits need no rebuild at all.** The dev bundle carries a `.devmode`
  marker that makes `bootstrap.py` start Streamlit with `--server.runOnSave` and
  the polling file watcher, so saving a file reloads the running app in ~2 s.

Rerun the script only after changing the Rust launcher or the dependency list.
The dev app has its own bundle identifier (`…mycol.dev`) and shows as
**Mycol (dev)**, so macOS never confuses it with a release build.

> [!NOTE]
> `poll` rather than the default watcher is deliberate: watchdog does not
> reliably see changes through the symlinked source tree.

### Why not `uv venv`

The interpreters are copied from uv's **managed** (python-build-standalone)
installs, not created with `uv venv`. A virtualenv only symlinks back to the
Python that built it, so a bundle made that way runs on the build machine and
nowhere else. A python-build-standalone copy references its own dylib through
`@rpath`, so `sys.prefix` follows the bundle wherever it goes. `verify_dist.sh`
asserts this, because the failure is invisible until someone else opens the app.

The copy also has its `EXTERNALLY-MANAGED` marker removed, or uv refuses to
install into it.

### Signing and notarization

Unsigned, the app is ad-hoc signed: it runs on the machine that built it, but
Gatekeeper blocks it anywhere else. For a distributable build, set both:

```bash
export MYCOL_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)"
export MYCOL_NOTARY_PROFILE="notarytool-profile"
./scripts/macos/make_dist.sh --version 0.2.0
```

This requires an Apple Developer Program membership. In CI the same values come
from the `MACOS_SIGN_IDENTITY` and `MACOS_NOTARY_PROFILE` secrets, with the
certificate in `MACOS_CERT_P12` / `MACOS_CERT_PASSWORD`. Without them the build
still succeeds and still produces a DMG — it just carries the Gatekeeper warning
documented in the README.

### macOS CI

[`.github/workflows/release-macos.yml`](../.github/workflows/release-macos.yml)
mirrors the Windows workflow: branch pushes build Apple Silicon only and upload
the DMG as an artifact; a `v*` tag builds both architectures on `macos-14`
(arm64) and `macos-13` (Intel) and attaches them to the same draft release.

# Linux

There is no Linux bundle. Run from source with `uv sync`.
