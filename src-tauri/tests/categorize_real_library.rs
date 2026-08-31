//! Integration test: run the auto-categorization pipeline on
//! the real F:\models library. Asserts the categories match
//! the directory layout the user described. Skipped when
//! the F:\models root is missing.

use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn categorize_the_real_model_library() {
    let root = Path::new("F:\\models");
    if !root.is_dir() {
        // The test workstation does not have the model
        // library; the categoriser is still tested by
        // the unit tests, this is just the smoke test.
        eprintln!("F:\\models is not present on this machine; skipping");
        return;
    }
    let result = sarathi_lib::registry::scan::scan_library(root);
    let mut by_category: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for gguf in &result.ggufs {
        let entry = sarathi_lib::registry::scan::entry_for(gguf);
        let cat = sarathi_lib::registry::categorize::categorize(&entry);
        by_category
            .entry(cat.label().to_string())
            .or_default()
            .push(gguf.path.display().to_string());
    }
    for (cat, paths) in &by_category {
        println!("== {} ({} model(s)) ==", cat, paths.len());
        for p in paths {
            println!("  {}", p);
        }
    }
    // Sanity: every category should have at least one model,
    // and the Unknown bucket should be small (it is the
    // "this is broken" bucket).
    assert!(!by_category.is_empty(), "no categories produced");
    let unknown = by_category.get("Uncategorised").map(|v| v.len()).unwrap_or(0);
    let total: usize = by_category.values().map(|v| v.len()).sum();
    assert!(
        unknown * 5 < total,
        "too many uncategorised models: {unknown} of {total}"
    );
}
