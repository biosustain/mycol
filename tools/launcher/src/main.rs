// Native launcher for the packaged Mycol bundle.
//
// It does one job: find the bundled interpreter and hand it bootstrap.py. The
// subsystem/console handling is what makes it worth a binary rather than a
// script — a release build shows no terminal, so failures have to surface in a
// dialog or the user sees nothing at all.
//
// Windowed-subsystem hint adapted from https://stackoverflow.com/a
// Posted by Freyja, modified by community. See post 'Timeline' for change history
// Retrieved 2026-01-14, License - CC BY-SA 4.0

#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{env, process::Command};

/// Where the bundle payload lives, relative to the launcher binary.
///
/// On Windows the payload sits beside the .exe. Inside a macOS .app the launcher
/// is at Contents/MacOS/Mycol and the payload is in Contents/Resources.
fn app_root(exe_path: &Path) -> PathBuf {
    let parent = exe_path.parent().expect("launcher has no parent directory");

    if cfg!(target_os = "macos") && parent.file_name() == Some(OsStr::new("MacOS")) {
        if let Some(contents) = parent.parent() {
            let resources = contents.join("Resources");
            if resources.is_dir() {
                return resources;
            }
        }
    }
    parent.to_path_buf()
}

fn python_path(app_root: &Path) -> PathBuf {
    if cfg!(windows) {
        app_root.join("bin/python_main/python.exe")
    } else {
        app_root.join("bin/python_main/bin/python")
    }
}

/// Show a message the user can actually see.
///
/// Release builds have no console on either platform, so a bare eprintln! goes
/// nowhere. Without this, a bundle that is missing files - most commonly because
/// it was run straight out of a zip without extracting - simply does nothing
/// when double-clicked.
#[cfg(target_os = "macos")]
fn show_error(message: &str) {
    let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        "display dialog \"{escaped}\" with title \"Mycol\" buttons {{\"OK\"}} default button \"OK\" with icon stop"
    );
    let _ = Command::new("osascript").arg("-e").arg(script).status();
}

#[cfg(target_os = "windows")]
fn show_error(message: &str) {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    // MessageBoxW lives in user32; without this the link fails.
    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(hwnd: *mut u8, text: *const u16, caption: *const u16, utype: u32) -> i32;
    }
    const MB_OK: u32 = 0x0000_0000;
    const MB_ICONERROR: u32 = 0x0000_0010;

    fn wide(s: &str) -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
    }

    let text = wide(message);
    let caption = wide("Mycol");
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), MB_OK | MB_ICONERROR);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn show_error(_message: &str) {}

fn fail(message: String) -> ! {
    eprintln!("{message}");
    show_error(&message);
    std::process::exit(1);
}

fn main() {
    let mut args: Vec<String> = env::args().collect();
    args.remove(0);

    let exe_path = match env::current_exe().and_then(|p| p.canonicalize()) {
        Ok(p) => p,
        Err(e) => fail(format!("Could not resolve the Mycol executable path.\n\n{e}")),
    };

    let root = app_root(&exe_path);

    let python = python_path(&root);
    if !python.exists() {
        fail(format!(
            "Mycol could not find its bundled Python.\n\n\
             Expected: {}\n\n\
             This usually means the app was launched without being fully \
             extracted, or part of the folder was moved. Extract the whole \
             download and keep its contents together.",
            python.display()
        ));
    }

    let bootstrap = root.join("bootstrap.py");
    if !bootstrap.exists() {
        fail(format!(
            "Mycol could not find bootstrap.py.\n\nExpected: {}\n\n\
             The installation looks incomplete - try extracting the download again.",
            bootstrap.display()
        ));
    }

    let mut binding = Command::new(&python);
    let cmd = binding.arg(&bootstrap).args(&args).current_dir(&root);

    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => fail(format!("Mycol failed to start its Python process.\n\n{e}")),
    }
}
