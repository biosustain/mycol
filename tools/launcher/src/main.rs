// Source - https://stackoverflow.com/a
// Posted by Freyja, modified by community. See post 'Timeline' for change history
// Retrieved 2026-01-14, License - CC BY-SA 4.0

#![cfg_attr(
  all(
    target_os = "windows",
    not(debug_assertions),
  ),
  windows_subsystem = "windows"
)]


use std::{env, process::Command, path::Path};

fn python_path(app_root: &Path) -> std::path::PathBuf {
    if cfg!(windows) {
        app_root.join("bin/python_main/python.exe")
    } else {
        app_root.join("bin/python_main/bin/python")
    }
}

fn main() {
    let mut args: Vec<String> = env::args().collect();
    let _program = args.remove(0);

    let exe_path = env::current_exe()
        .and_then(|p| p.canonicalize())
        .expect("Failed to resolve executable path");

    let app_root = exe_path.parent().expect("Missing app root");

    let python = python_path(app_root);
    if !python.exists() {
        eprintln!("Python not found: {}", python.display());
        std::process::exit(1);
    }

    let bootstrap = app_root.join("bootstrap.py");
    if !bootstrap.exists() {
        eprintln!("bootstrap.py not found: {}", bootstrap.display());
        std::process::exit(1);
    }

    
    let mut binding = Command::new(python);
    let cmd = binding.arg(bootstrap).args(args);

    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }


    let status = cmd
        .status()
        .expect("Failed to execute Python");

    std::process::exit(status.code().unwrap_or(1));
}
