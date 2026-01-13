#![windows_subsystem = "windows"]

use std::env;
use std::process::Command;

fn main() {
    let exe_path = env::current_exe().expect("Failed to get current executable path");
    let app_root = exe_path.parent().expect("Failed to get app root");

    let python_path = if cfg!(target_os = "windows") {
        app_root.join("bin").join("python_main").join("python.exe")
    } else {
        app_root.join("bin").join("python_main").join("bin").join("python")
    };

    if !python_path.exists() {
        eprintln!("Error: Python not found at {:?}", python_path);
        std::process::exit(1);
    }

    let bootstrap_path = app_root.join("bootstrap.py");
    if !bootstrap_path.exists() {
        eprintln!("Error: bootstrap.py not found at {:?}", bootstrap_path);
        std::process::exit(1);
    }

    let args: Vec<String> = env::args().skip(1).collect();

    #[cfg(target_os = "windows")]
    use std::os::windows::process::CommandExt;

    // CREATE_NO_WINDOW = 0x08000000
    #[cfg(target_os = "windows")]
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let mut cmd = Command::new(python_path);
    cmd.arg(bootstrap_path).args(args);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let status = cmd.status().expect("Failed to execute Python process");

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}
