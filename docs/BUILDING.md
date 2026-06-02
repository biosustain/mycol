## Building

> [!NOTE]
> Building requires the rust toolchain to be installed. 

To "build" run

```bash
./scripts/make_dist.sh
```

or on windows

```ps1
./scripts/make_dist.ps1
```

note that you can only distribute the app on the same os it was built on



### How does this work and what does it do?
This script does a few things:

1. It downloads the corresponding python binaries (3.13.1 and 3.10.12) for the current OS (Windows for powershell and Linux for bash) and puts it in the `dist/app/bin` folder.
2. It installs the dependencies for both python versions using uv. This means that the two binaries are (in theory)self-contained and do not require any additional dependencies.
3. It copies the source code to the `dist/app/src` folder.
4. It compiles the rust launcher from the `tools/launcher` folder and puts it in the `dist/app` folder. -> this is literally just a wrapper around the python binary that calls the `bootstrap` script, so that we can have an actual executable on the target system.


### Why is it so big? 
We are bundeling two complete versions of python (3.13 and 3.10) + cuda support on windows.

### How do I install and build the launcher?


Install rust using [rustup](https://rust-lang.org/tools/install/)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

then using rustup install the toolchain

```bash
rustup install stable
```

the launcher will be build as part of the `make_dist` script and copied to the correct version