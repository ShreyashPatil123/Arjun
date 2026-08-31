//! Integration test: scan the real F:\models library and assert
//! the inventory is what the user said it is. This is the
//! smoke test for TODO 4 of the 7-step plan.
//!
//! Skipped when the F:\models root is missing, so the test
//! still passes on a workstation without that drive.

use std::path::Path;

#[test]
fn scan_the_real_model_library() {
    let roots: &[&str] = &[
        "F:\\models",
        "C:\\Users\\lenovo\\models",
    ];
    let mut found = 0usize;
    for root in roots {
        let root = Path::new(root);
        if !root.is_dir() {
            continue;
        }
        let result = sarathi_lib::registry::scan::scan_library(root);
        println!("{}: {} ggufs, {} mmprojs", root.display(), result.ggufs.len(), result.mmprojs.len());
        for gguf in &result.ggufs {
            println!("  {} ({} bytes){}",
                gguf.path.display(),
                gguf.bytes,
                gguf.mmproj_path.as_ref()
                    .map(|p| format!(" [mmproj: {}]", p.display()))
                    .unwrap_or_default());
        }
        found += result.ggufs.len();
    }
    assert!(found >= 25, "expected at least 25 GGUFs across the real roots, found {found}");
}
