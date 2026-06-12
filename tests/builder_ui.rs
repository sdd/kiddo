use std::env;
use std::path::{Path, PathBuf};

struct CurrentDirGuard(PathBuf);

impl CurrentDirGuard {
    fn to_manifest_dir() -> Self {
        let original = env::current_dir().unwrap();
        env::set_current_dir(env!("CARGO_MANIFEST_DIR")).unwrap();
        Self(original)
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.0);
    }
}

fn ui_test_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[ignore] // temporarily ignore until I can get this test working in CI with cargo nextest
#[test]
fn periodic_queries_cannot_return_points() {
    let _cwd = CurrentDirGuard::to_manifest_dir();
    let t = trybuild::TestCases::new();
    t.compile_fail(ui_test_path("tests/ui/periodic_with_points.rs"));
}
