// The crate embeds its WGSL shaders at compile time via `load_internal_asset!`
// (`include_str!`). cargo does not reliably treat those `.wgsl` files as build
// inputs, so editing only a shader would NOT recompile the crate and the change
// would silently never reach the binary. Emit an explicit rerun-if-changed for
// every shader so any shader edit invalidates the build.
use std::path::Path;

fn main() {
    let shader_dir = Path::new("src/shaders");
    // Watch the directory itself (catches added/removed files) ...
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    // ... and each individual shader file (catches in-place edits).
    if let Ok(entries) = std::fs::read_dir(shader_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    // Always re-run if the build script itself changes.
    println!("cargo:rerun-if-changed=build.rs");
}
